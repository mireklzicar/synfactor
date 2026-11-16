"""Post-hoc full-Freedom Dock12 evaluation against frozen Orion archives.

Run with .venv-eval. All charged history rows, including warm starts and failed
calls, count against budget. Portfolios require valid docking status, finite
score and valid original SMILES. Druglike-v1 is applied to that original form,
without parent standardization, neutralization or tautomer processing.

Orion compatibility: Morgan radius2/2048/no chirality, STRICT Tc < .30, k100.
Canonical isomeric molecular duplicates retain the best-score route. This
preserves the strict-fingerprint portfolio optimum; ordinary top-N is explicitly
molecular, unlike the historical route-level ordinary top-N tables.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ORION = Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2_orion')
BENCHMARK = Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2')
PRODUCTION = Path('/home/mirekl/DeepMedChem_Prod/dmc-navigator-prod/src/dmc_navigator_prod/core/filters.py')
PRODUCTION_SHA256 = '28a01da8a9da1a87b43604363fd044294a3c1f5617e48e167c1f302f3969b3c1'
METHODS = ('ga_dcso_v14_screening', 'spacehasten_fast', 'spacehasten_accurate')
ALIASES = {x: x+'_spacehasten' for x in ('kif11', 'pyrd', 'tgfr1')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as handle:
        return json.load(handle)


def production_gate(path=PRODUCTION):
    if sha(path) != PRODUCTION_SHA256:
        raise ValueError('Production gate hash changed; audit before changing the frozen gate contract')
    name = '_synfactor_orion_frozen_gate'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ExactFilterGate(module.get_filter_profile('druglike-v1'))


def read_history(path, expected_calls=10000, expected_warm=1000):
    payload = load_json(path)
    metadata = payload if isinstance(payload, dict) else {}
    rows = payload if isinstance(payload, list) else payload.get('history', payload.get('rows'))
    if not isinstance(rows, list) or len(rows) != expected_calls:
        raise ValueError(f'Need exactly {expected_calls} charged history rows, including failures/warm starts')
    if any(not isinstance(r.get('product_id'), str) or not r['product_id'] for r in rows):
        raise ValueError('Every charged row needs a product_id')
    if len({r['product_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate charged product IDs; do not silently deduplicate the budget')
    if any('status' not in r or 'iteration' not in r for r in rows):
        raise ValueError('Every charged row requires status and iteration')
    warm = sum(int(r['iteration']) < 0 for r in rows)
    if warm != expected_warm:
        raise ValueError(f'Need {expected_warm} charged warm rows (iteration=-1), got {warm}')
    return rows, metadata


def warm_evidence(repo, target, seed):
    """Reconstruct the charged initialization from original, hashable sources."""
    cohort = repo/f'artifacts/seeds/freedom_296b/seed_{seed}/selected_10k/subselections'
    ids_path = cohort/'budget_01000.ids'
    candidates_path = cohort/'budget_01000_candidates.csv'
    matrix = repo/f'artifacts/dock12_seed_screen/freedom/seed_{seed}/results_long.csv'
    ids = ids_path.read_text().splitlines()
    with candidates_path.open() as handle:
        candidates = {r['product_id']: r['smiles'] for r in csv.DictReader(handle)}
    wanted = set(ids)
    with matrix.open() as handle:
        selected = [r for r in csv.DictReader(handle)
                    if r['target_id'] == target and r['product_id'] in wanted]
    scores = {r['product_id']: r for r in selected}
    if len(ids) != 1000 or len(wanted) != 1000 or len(selected) != 1000 or set(scores) != wanted:
        raise ValueError('Invalid/missing shared warm source cohort')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['product_id', 'status', 'score'])
    writer.writeheader()
    rows = []
    for pid in ids:
        raw = scores[pid]
        valid = raw['valid'].lower() in ('true', '1', 'yes')
        # Hash the original strings exactly as the benchmark warm CSV does.
        writer.writerow(dict(product_id=pid, status='valid' if valid else 'failed',
                             score=raw['score'] if valid else ''))
        score = float(raw['score']) if valid else None
        if score is not None and (not np.isfinite(score) or abs(score) >= 100):
            valid, score = False, None
        rows.append(dict(product_id=pid, smiles=candidates[pid], score=score,
                         status='valid' if valid else 'failed', iteration=-1,
                         source='charged_shared_warm', charged=True))
    return rows, dict(ids_sha256=sha(ids_path), candidates_sha256=sha(candidates_path),
                     matrix_sha256=sha(matrix),
                     csv_sha256=hashlib.sha256(buffer.getvalue().encode()).hexdigest())


def verify_comparison_identity(history_path, rows, target, seed, benchmark_repo,
                               grid_assertion=None, warm_assertion=None):
    """Bind comparison identity to the warm sources and each charged response.

    CLI hashes are assertions, never substitutes for evidence. A copied standalone
    history can still be evaluated with --skip-baselines, without private sources.
    """
    root = history_path.parent
    config_path = root/'config.json'
    if not config_path.exists():
        raise ValueError('Matched comparison requires adjacent config.json and batch evidence; use --skip-baselines for standalone evaluation')
    config = load_json(config_path)
    if ALIASES.get(config.get('target'), config.get('target')) != target or config.get('seed') != seed:
        raise ValueError('History configuration target/seed does not match requested comparison')
    if (config.get('budget'), config.get('warm_charged'), config.get('batch_size')) != (10000, 1000, 1000):
        raise ValueError('History configuration does not match the 10000/1000/1000 budget')
    grid, warm_hash = config.get('grid_sha256'), config.get('warm', {}).get('csv_sha256')
    if not grid or not warm_hash:
        raise ValueError('Missing grid/warm hashes in history configuration')
    if grid_assertion is not None and grid_assertion != grid:
        raise ValueError('CLI grid hash differs from history configuration')
    if warm_assertion is not None and warm_assertion != warm_hash:
        raise ValueError('CLI warm hash differs from history configuration')
    expected_warm, hashes = warm_evidence(benchmark_repo, config['target'], seed)
    if hashes != config['warm'] or rows[:1000] != expected_warm:
        raise ValueError('History warm rows/source hashes differ from shared benchmark initialization')
    def protocol_hash(protocol):
        identity = {k: v for k, v in protocol.items() if k not in ('free_disk_gib', 'preflight_only')}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    def check_protocol(protocol):
        if (protocol_hash(protocol) != config.get('oracle_protocol_sha256')
                or protocol.get('grid_sha256') != grid or protocol.get('target') != config['target']
                or protocol.get('seed') != seed):
            raise ValueError('Charged oracle protocol differs from pinned target/seed/grid')
    check_protocol(load_json(root/'remote_preflight.json'))
    evidence = {'config_sha256': sha(config_path), 'warm_sources': hashes, 'batch_response_sha256': {}}
    for iteration in range(1, 10):
        folder = root/f'batch_{iteration:02d}'
        proposal = load_json(folder/'proposal.json')
        response = load_json(folder/'normalized.json')
        request_hash = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
        if (response.get('request_sha256') != request_hash or response.get('smoke_only')
                or proposal.get('target') != config['target'] or proposal.get('seed') != seed
                or proposal.get('batch_id') != folder.name
                or any(response.get(k) != proposal.get(k) for k in ('target', 'seed', 'run_id', 'batch_id'))):
            raise ValueError('Charged batch request/response identity mismatch')
        check_protocol(response['protocol'])
        observed = [dict(product_id=r['product_id'], smiles=r['smiles'], score=r['score'],
                         status='valid' if r['valid'] else 'failed', iteration=iteration,
                         source='live_glide', charged=True) for r in response['rows']]
        if (len(observed) != 1000 or [r['product_id'] for r in observed] != proposal['product_ids']
                or observed != rows[iteration*1000:(iteration+1)*1000]):
            raise ValueError('History observations differ from charged batch response')
        evidence['batch_response_sha256'][folder.name] = sha(folder/'normalized.json')
    return grid, warm_hash, evidence


def molecular_rows(rows, gate, *, statuses_required=True):
    valid, failures = [], Counter()
    for row in rows:
        if statuses_required and row.get('status') != 'valid':
            failures['docking_status_not_valid'] += 1
            continue
        try:
            score = float(row['score'])
        except (KeyError, TypeError, ValueError):
            score = float('nan')
        if not np.isfinite(score):
            failures['nonfinite_score'] += 1
            continue
        smiles = row.get('smiles')
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            failures['invalid_smiles'] += 1
            continue
        # Gate must see original form, not canonicalized/standardized SMILES.
        reason = gate.evaluate(smiles)
        valid.append(dict(product_id=row['product_id'], smiles=smiles,
            canonical_smiles=Chem.MolToSmiles(mol, isomericSmiles=True), score=score,
            iteration=int(row.get('iteration', -1)), druglike_failure=reason))
    valid.sort(key=lambda r: (r['score'], r['product_id']))
    seen, unique = set(), []
    for row in valid:
        if row['canonical_smiles'] not in seen:
            seen.add(row['canonical_smiles']); unique.append(row)
    return unique, dict(charged_input_rows=len(rows), valid_route_rows=len(valid),
        unique_valid_molecules=len(unique), duplicate_molecules_removed=len(valid)-len(unique),
        exclusions=dict(failures), druglike_failures=dict(Counter(r['druglike_failure'] for r in unique if r['druglike_failure'])))


def strict_conflicts(smiles, threshold=.30):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    graph = np.zeros((len(fps), len(fps)), dtype=bool)
    for i, fp in enumerate(fps):
        # Equality is a conflict: Orion requires Tc strictly BELOW .30.
        conflict = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps[i+1:])) >= threshold
        graph[i, i+1:] = conflict; graph[i+1:, i] = conflict
    return graph


def saved_opdiv(path, archive_path, source_rows, archive, graph, gate_config, mode):
    """Reuse only verified compatible bounds; otherwise return a reason to refit."""
    if path is None or not path.exists():
        return None, 'no_saved_cell'
    saved = load_json(path)
    if mode == 'druglike-v1':
        config = saved.get('config', {})
        if (saved.get('metric') != 'morgan' or config.get('profile') != 'druglike-v1'
                or config.get('production_sha256') != PRODUCTION_SHA256
                or saved.get('archive_sha256') != sha(archive_path)):
            return None, 'gate_or_archive_hash_mismatch'
        portfolio = saved.get('portfolio', {})
    else:
        stripped = [{k: v for k, v in r.items() if k != 'properties'} for r in source_rows]
        digest = hashlib.sha256(json.dumps(stripped, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if saved.get('archive_rows_sha256') != digest or saved.get('config', {}).get('threshold') != .3:
            return None, 'unfiltered_archive_or_threshold_mismatch'
        portfolio = saved.get('optimized', {})
    # Version equality protects fingerprint, canonical identity and descriptor
    # semantics; a source hash alone cannot establish cross-RDKit parity.
    if gate_config.get('rdkit') != rdBase.rdkitVersion:
        return None, 'rdkit_version_mismatch'
    if saved.get('config', {}).get('k') != 100 or int(saved.get('charged', 0)) != 10000:
        return None, 'cardinality_or_budget_mismatch'
    selected = portfolio.get('selected', [])
    if len(selected) != 100:
        return None, 'saved_portfolio_not_100'
    mapping = {r['canonical_smiles']: i for i, r in enumerate(archive)}
    try:
        ids = [mapping[Chem.MolToSmiles(Chem.MolFromSmiles(r['smiles']), isomericSmiles=True)] for r in selected]
    except (KeyError, TypeError, ValueError):
        return None, 'saved_portfolio_members_not_in_current_archive'
    if len(set(ids)) != 100 or graph[np.ix_(ids, ids)].any():
        return None, 'saved_portfolio_not_feasible_under_current_fingerprints'
    mean = float(np.mean([archive[i]['score'] for i in ids]))
    if abs(mean-float(portfolio['mean'])) > 1e-8:
        return None, 'canonical_representative_score_mismatch'
    solver = saved['solver']; lower = -mean; upper = -float(solver['bound'])
    if upper < lower-1e-7:
        raise ValueError('Invalid historical optimization interval')
    return dict(metric='OPDiv', value=lower, raw_mean=mean, feasible_lower_bound=lower,
        upper_bound=max(lower, upper), gap=max(0., upper-lower), status=solver['status'],
        certified=bool(solver['certified']), selected_count=100,
        source='verified historical bound', source_path=str(path), source_sha256=sha(path)), None


def evaluate_domain(rows, max_seconds, *, saved_path=None, archive_path=None,
                    source_rows=None, gate_config=None, mode='all-valid'):
    from opdiv import select
    scores = np.array([r['score'] for r in rows], dtype=float)
    result = dict(eligible_molecules=len(rows), scores={}, portfolios=[])
    for k in (100, 1000):
        result['scores'][f'molecular_top{k}_count'] = min(k, len(rows))
        result['scores'][f'molecular_top{k}_mean'] = float(scores[:k].mean()) if len(rows) >= k else None
    if len(rows) < 100:
        result['portfolios'] = [dict(metric=m, status='insufficient_candidates', selected_count=0,
            value=None, raw_mean=None, feasible_lower_bound=None, upper_bound=None, certified=False) for m in ('GPDiv', 'OPDiv')]
        return result
    started = time.perf_counter()
    graph = strict_conflicts([r['smiles'] for r in rows])
    result['graph_seconds'] = time.perf_counter()-started
    old, reason = saved_opdiv(saved_path, archive_path, source_rows, rows, graph, gate_config, mode)
    result['historical_bound_not_reused_reason'] = reason
    for name, method in [('GPDiv', 'greedy'), ('OPDiv', 'optimal')]:
        if name == 'OPDiv' and old is not None:
            result['portfolios'].append(old); continue
        begin = time.perf_counter()
        choice = select(-scores, k=100, conflicts=graph, method=method,
                        **({'time_limit': max_seconds} if method == 'optimal' else {}))
        complete = len(choice.indices) == 100
        value = float(choice.value) if complete and choice.value is not None else None
        result['portfolios'].append(dict(metric=name, status=choice.status,
            selected_count=len(choice.indices), value=value, raw_mean=-value if value is not None else None,
            feasible_lower_bound=value, upper_bound=choice.upper_bound,
            certified=method=='optimal' and choice.status=='optimal',
            source='recomputed with current strict fingerprint contract', wall_seconds=time.perf_counter()-begin,
            selected_product_ids=[rows[int(i)]['product_id'] for i in choice.indices]))
    return result


def baseline_cells(orion, target, seed):
    refs = pd.read_csv(orion/'three_seed_revision/source_cohort/cell_metrics.csv')
    refs = refs[(refs.target_id == target) & (refs.seed == seed) & refs.method.isin(METHODS)]
    if len(refs) != len(METHODS) or set(refs.method) != set(METHODS):
        raise ValueError(f'Expected exactly {METHODS} for {target}, seed{seed}')
    for reference in refs.to_dict('records'):
        name = reference['run_id']
        roots = [orion/'three_seed_revision/druglike_v1', orion/'druglike_v1']
        source = next((r/'archives'/f'{name}.json.gz' for r in roots if (r/'archives'/f'{name}.json.gz').exists()), None)
        if source is None:
            source = orion/'medchem_baseline/archives'/f'{name}.json.gz'
        payload = load_json(source)
        if (int(reference['charged']) != 10000 or len(payload['rows']) != int(reference['valid'])
                or payload.get('reference', {}).get('run_id') != name):
            raise ValueError(f'Baseline archive/budget identity mismatch for {name}')
        filtered = source.parent.parent/'cells'/f'{name}__morgan.json'
        unfiltered = orion/'three_seed_revision/unfiltered_morgan/refined_cells'/f'{name}.json'
        yield reference, source, payload, {'all-valid':unfiltered, 'druglike-v1':filtered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--grid-sha256')
    parser.add_argument('--warm-sha256')
    parser.add_argument('--orion', type=Path, default=ORION)
    parser.add_argument('--benchmark-repo', type=Path, default=BENCHMARK,
                        help='Original warm IDs, candidate SMILES and score matrix for matched comparisons')
    parser.add_argument('--production-filter', type=Path, default=PRODUCTION)
    parser.add_argument('--max-seconds', type=float, default=30)
    parser.add_argument('--skip-baselines', action='store_true')
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error('max-seconds must be positive')
    target = ALIASES.get(args.target, args.target)
    rows, metadata = read_history(args.history)
    config = metadata.get('config', {})
    grid = args.grid_sha256 or metadata.get('grid_sha256') or config.get('grid_sha256')
    warm = args.warm_sha256 or metadata.get('warm_sha256') or config.get('warm_sha256')
    identity_evidence = None
    if not args.skip_baselines:
        grid, warm, identity_evidence = verify_comparison_identity(
            args.history, rows, target, args.seed, args.benchmark_repo,
            args.grid_sha256, args.warm_sha256)
    gate = production_gate(args.production_filter)
    baselines = [] if args.skip_baselines else list(baseline_cells(args.orion, target, args.seed))
    for reference, *_ in baselines:
        if grid and reference['grid_sha256'] != grid:
            raise ValueError(f'Grid mismatch for {reference["run_id"]}; BRD4 must use repaired3MXF/JQ1 BD1')
        if warm and reference['warm_sha256'] != warm:
            raise ValueError(f'Warm-start evidence hash mismatch for {reference["run_id"]}')
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(target=target, seed=args.seed, charged_calls=len(rows), charged_warm=1000,
        history_path=str(args.history), history_sha256=sha(args.history), rdkit=rdBase.rdkitVersion,
        gate_path=str(args.production_filter), gate_sha256=PRODUCTION_SHA256,
        fingerprint='Morgan radius2/2048/no chirality; conflicts Tc>=0.30; portfolios STRICT Tc<0.30',
        canonical_duplicates='best score; ties by product_id; original SMILES retained for gate/fingerprints',
        eligibility='status valid, finite score and valid original SMILES; failures remain charged',
        grid_sha256=grid, warm_sha256=warm, grid_verified_against_cohort=bool(grid and baselines),
        warm_verified_against_cohort=bool(warm and baselines),
        comparison_verified=bool(grid and warm and baselines),
        identity_evidence=identity_evidence,
        grid_note='KIF11/PYRD/TGFR1 legacy continuity *_spacehasten grids; BRD4 repaired3MXF/JQ1 BD1',
        solver_time_limit_seconds=args.max_seconds, cells=[])
    cells = [('synfactor', rows, None, None, None, {})]
    cells += [(ref['method'], payload['rows'], source, payload.get('config', {}), saved, ref)
              for ref, source, payload, saved in baselines]
    began = time.perf_counter()
    for method, input_rows, source, gate_config, saved, reference in cells:
        unique, audit = molecular_rows(input_rows, gate, statuses_required=method=='synfactor')
        audit['charged_calls'] = len(rows) if method=='synfactor' else int(reference['charged'])
        cell = dict(method=method, reference=reference, audit=audit, domains={})
        if source is not None:
            cell.update(archive_path=str(source), archive_sha256=sha(source))
        for mode in ('all-valid', 'druglike-v1'):
            eligible = unique if mode=='all-valid' else [r for r in unique if r['druglike_failure'] is None]
            cell['domains'][mode] = evaluate_domain(eligible, args.max_seconds,
                saved_path=saved.get(mode) if saved else None, archive_path=source,
                source_rows=input_rows, gate_config=gate_config, mode=mode)
            pd.DataFrame(eligible).to_csv(args.out/f'{method}__{mode}__molecules.csv.gz', index=False)
        summary['cells'].append(cell)
        summary['wall_seconds'] = time.perf_counter()-began
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(f'Evaluated {method}: {len(unique)} unique valid molecules', flush=True)
    flat = []
    for cell in summary['cells']:
        for mode, domain in cell['domains'].items():
            row = dict(method=cell['method'], domain=mode, charged_calls=cell['audit']['charged_calls'],
                       eligible_molecules=domain['eligible_molecules'], **domain['scores'])
            for portfolio in domain['portfolios']:
                row.update({f'{portfolio["metric"]}_{k}':portfolio.get(k) for k in
                            ('value','raw_mean','status','feasible_lower_bound','upper_bound','certified')})
            flat.append(row)
    pd.DataFrame(flat).to_csv(args.out/'comparison.csv', index=False)


if __name__ == '__main__':
    main()
