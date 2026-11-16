"""All-valid TYK2 comparisons at identical charged 1K-round budgets.

Default evaluation is raw molecular scores plus fast strict-Tc GPDiv100.
--optimal additionally invokes Orion's full-domain SCIP independent-set solver.
No druglike filters or oracle calls are performed. Failed calls remain charged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .live_evaluate import (BENCHMARK, ORION, baseline_cells, load_json,
                            molecular_rows, sha, strict_conflicts, warm_evidence)


class AllowAll:
    def evaluate(self, smiles):
        return None


def checked_prefix(rows, rounds, *, adaptive_offset):
    """Check a charged ledger, never infer budget from valid-only archives."""
    if not 0 <= rounds <= 9:
        raise ValueError('Adaptive rounds must be between zero and nine')
    if len(rows) < 1000*(rounds+1) or len(rows) > 10000 or len(rows) % 1000:
        raise ValueError('Ledger must contain complete 1000-call rounds through requested budget')
    if len({row['product_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate charged identities')
    completed = len(rows)//1000-1
    expected = {-1:1000, **{i+adaptive_offset:1000 for i in range(completed)}}
    if Counter(int(row['iteration']) for row in rows) != expected:
        raise ValueError('Charged iteration coverage/count differs from complete 1000-call rounds')
    prefix = [row for row in rows if int(row['iteration']) == -1
              or adaptive_offset <= int(row['iteration']) < rounds+adaptive_offset]
    if len(prefix) != 1000*(rounds+1):
        raise ValueError('Prefix charged count mismatch')
    return prefix


def core_row(row):
    return {key:row.get(key) for key in ('product_id','smiles','score','status','iteration','charged')}


def protocol_hash(protocol):
    identity = {k:v for k,v in protocol.items() if k not in ('free_disk_gib','preflight_only')}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def verify_partial(path, rounds, benchmark_repo, target='tyk2', seed=0):
    root = path.parent
    payload = load_json(path)
    rows = checked_prefix(payload['history'], rounds, adaptive_offset=1)
    config = load_json(root/'config.json')
    if config.get('target') != target or config.get('seed') != seed:
        raise ValueError('State target/seed differs from requested comparison')
    if tuple(config.get(k) for k in ('budget','warm_charged','batch_size')) != (10000,1000,1000):
        raise ValueError('Expected frozen 10000/1000/1000 budget configuration')
    warm, hashes = warm_evidence(benchmark_repo,target,seed)
    if hashes != config.get('warm') or [core_row(r) for r in rows[:1000]] != [core_row(r) for r in warm]:
        raise ValueError('Warm source hashes or charged observations differ')
    def check(protocol):
        if (protocol_hash(protocol) != config.get('oracle_protocol_sha256')
                or protocol.get('grid_sha256') != config.get('grid_sha256')
                or protocol.get('target') != target or protocol.get('seed') != seed):
            raise ValueError('Oracle protocol target/seed/grid/hash mismatch')
    check(load_json(root/'remote_preflight.json'))
    lineage=None
    if (root/'lineage.json').exists():
        from .tyk2_branch import verify_lineage, verify_run_bindings
        lineage=verify_lineage(root,payload,config)
        verify_run_bindings(root,config,rounds,lineage)
    evidence = dict(state_sha256=sha(path), config_sha256=sha(root/'config.json'),
                    warm_sources=hashes, response_sha256={})
    if lineage is not None:
        evidence['lineage']=lineage
    for iteration in range(1,rounds+1):
        folder=root/f'batch_{iteration:02d}'
        proposal,response=load_json(folder/'proposal.json'),load_json(folder/'normalized.json')
        expected=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()
        if (response.get('request_sha256') != expected or response.get('smoke_only')
                or proposal.get('target') != target or proposal.get('seed') != seed
                or proposal.get('batch_id') != folder.name
                or any(response.get(k) != proposal.get(k) for k in ('target','seed','run_id','batch_id'))):
            raise ValueError('Batch request/response identity mismatch')
        check(response['protocol'])
        observed=[dict(product_id=r['product_id'],smiles=r['smiles'],score=r['score'],
                       status='valid' if r['valid'] else 'failed',iteration=iteration,charged=True)
                  for r in response['rows']]
        actual=[core_row(r) for r in rows[iteration*1000:(iteration+1)*1000]]
        if (len(observed)!=1000 or observed!=actual
                or [r['product_id'] for r in observed]!=proposal.get('product_ids')):
            raise ValueError('Charged observations differ from immutable batch evidence')
        evidence['response_sha256'][folder.name]=sha(folder/'normalized.json')
    return rows,config,evidence


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def baseline_prefix(reference, archive, rounds, benchmark_repo, config, warm):
    """Bind valid archive rows to original full charged ledgers and manifests."""
    name=reference['run_id']
    if (reference['grid_sha256']!=config['grid_sha256']
            or reference['warm_sha256']!=config['warm']['csv_sha256']):
        raise ValueError('Baseline reference grid/warm differs')
    is_ga=reference['method']=='ga_dcso_v14_screening'
    root=benchmark_repo/'artifacts'/('navigator_glide' if is_ga else 'spacehasten_glide')/name
    manifest=load_json(root/'run_manifest.json')
    evidence={'manifest_sha256':sha(root/'run_manifest.json'),'ledger_sha256':{}}
    if is_ga:
        if (manifest['target_id']!=config['target'] or manifest['search_seed']!=config['seed']
                or manifest['grid']['sha256']!=config['grid_sha256']
                or manifest['warm_start']['sha256']!=config['warm']['csv_sha256']
                or manifest['budget']['submitted']!=10000 or manifest['budget']['batch_size']!=1000
                or manifest['warm_start']['n']!=1000 or manifest['warm_start']['ids_sha256']!=config['warm']['ids_sha256']):
            raise ValueError('GA manifest identity/budget mismatch')
        warm_path=root/'warm_start_scores.csv'
        if sha(warm_path)!=config['warm']['csv_sha256']:
            raise ValueError('GA warm CSV hash mismatch')
        ledger=[]
        for iteration,path in [(-1,warm_path)]+[(i,root/f'oracle_scores/iteration_{i:04d}_handoff.csv') for i in range(9)]:
            values=read_csv(path)
            if len(values)!=1000:
                raise ValueError('GA original handoff does not charge 1000 rows')
            for row in values:
                ledger.append(dict(product_id=row['product_id'],status=row['status'],
                                   score=float(row['score']) if row['score'] else None,iteration=iteration))
            evidence['ledger_sha256'][str(path)]=sha(path)
    else:
        cell=manifest['cell']
        if (cell['target_id']!=config['target'] or cell['seed']!=config['seed']
                or cell['grid_sha256']!=config['grid_sha256']
                or cell['warm']['sha256']!=config['warm']['csv_sha256']
                or cell['warm']['ids_sha256']!=config['warm']['ids_sha256']
                or tuple(cell[k] for k in ('budget','warm_charged','batch_size','adaptive_batches'))!=(10000,1000,1000,9)):
            raise ValueError('SpaceHASTEN manifest identity/budget mismatch')
        ledger=load_json(root/'state.json')['history']
        evidence['ledger_sha256'][str(root/'state.json')]=sha(root/'state.json')
    prefix=checked_prefix(ledger,rounds,adaptive_offset=0)
    if len(ledger)!=10000:
        raise ValueError('Archived baseline must have its full 10000-call ledger')
    warm_actual=[r for r in ledger if int(r['iteration'])==-1]
    keys=('product_id','status','score')
    if [{k:r.get(k) for k in keys} for r in warm_actual]!=[{k:r.get(k) for k in keys} for r in warm]:
        raise ValueError('Baseline actual warm observations differ from shared cohort')
    all_valid={r['product_id']:r for r in ledger if r.get('status')=='valid'}
    if len({r['product_id'] for r in archive})!=len(archive) or set(all_valid)!={r['product_id'] for r in archive}:
        raise ValueError('Valid-only archive does not match original charged ledger')
    for row in archive:
        old=all_valid[row['product_id']]
        if int(row['iteration'])!=int(old['iteration']) or float(row['score'])!=float(old['score']):
            raise ValueError('Archived score/iteration differs from original ledger')
    wanted={r['product_id'] for r in prefix}
    return [r for r in archive if r['product_id'] in wanted],evidence


def greedy_portfolio(rows, k=100):
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    selected,fps=[],[]
    for index,row in enumerate(rows):
        fp=generator.GetFingerprint(Chem.MolFromSmiles(row['smiles']))
        if not fps or max(DataStructs.BulkTanimotoSimilarity(fp,fps))<.30:
            selected.append(index);fps.append(fp)
            if len(selected)==k:
                break
    value=-float(np.mean([rows[i]['score'] for i in selected])) if len(selected)==k else None
    return dict(metric='GPDiv',selected_count=len(selected),value=value,
                feasible_lower_bound=value,upper_bound=None,certified=False,
                raw_mean=-value if value is not None else None,status='greedy' if value is not None else 'insufficient_greedy_cardinality',
                selected_product_ids=[rows[i]['product_id'] for i in selected])


def optimal_portfolio(rows, benchmark_repo, seconds):
    started=time.perf_counter()
    sys.path.insert(0,str(benchmark_repo/'scripts'))
    source=importlib.import_module('analyze_diversity_portfolios')
    scores=np.asarray([r['score'] for r in rows],dtype=float)
    graph=strict_conflicts([r['smiles'] for r in rows])
    adj=[int.from_bytes(np.packbits(row,bitorder='little').tobytes(),'little') for row in graph]
    active,removed=source.dominate(adj,scores)
    starts=[]
    for scale in (0.,.1,.5):
        for seed in range(3):
            order=np.lexsort((np.arange(len(rows)),scores+np.random.default_rng(seed).gumbel(size=len(rows))*scale))
            chosen=source.greedy(adj,order,active,k=100)
            if len(chosen)==100:
                starts.append(chosen)
    if not starts:
        return dict(metric='OPDiv',status='no_feasible_100_start',value=None,upper_bound=None,certified=False)
    preprocess=time.perf_counter()-started
    result=source.solve_independent(adj,scores,active,starts,seconds,k=100)
    source.validate_selection(adj,result['selected'],k=100)
    value=-float(scores[result['selected']].mean())
    return dict(metric='OPDiv',status=result['status'],value=value,raw_mean=-value,
                feasible_lower_bound=value,upper_bound=max(value,-float(result['bound'])),
                gap=result['gap'],certified=result['certified'],selected_count=100,
                selected_product_ids=[rows[i]['product_id'] for i in result['selected']],
                source_sha256=sha(Path(source.__file__)),full_domain=True,
                exact_dominance_removed=len(rows)-active.bit_count(),preprocess_seconds=preprocess,
                solver_seconds=result['solver_seconds'],wall_seconds=time.perf_counter()-started)


def evaluate(rows, charged, statuses_required, optimal, seconds, benchmark_repo):
    unique,audit=molecular_rows(rows,AllowAll(),statuses_required=statuses_required)
    audit['charged_calls']=charged
    scores=np.asarray([r['score'] for r in unique])
    result=dict(audit=audit,eligible_molecules=len(unique),scores={},portfolios=[])
    for k in (1,10,100,1000):
        result['scores'][f'top{k}_raw_mean']=float(scores[:k].mean()) if len(scores)>=k else None
    result['portfolios'].append(greedy_portfolio(unique))
    if optimal:
        result['portfolios'].append(optimal_portfolio(unique,benchmark_repo,seconds))
    return result,unique


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--charged',type=int,help='Default: all complete rounds currently in history')
    p.add_argument('--target',default='tyk2');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--previous',type=Path,required=True,help='Previous SynFactor v1 state.json; same prefix evaluated')
    p.add_argument('--benchmark-repo',type=Path,default=BENCHMARK)
    p.add_argument('--orion',type=Path,default=ORION)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--optimal',action='store_true')
    p.add_argument('--max-seconds',type=float,help='SCIP solver cap per method; default3s partial or120s full')
    args=p.parse_args()
    charged=args.charged or len(load_json(args.history)['history'])
    if not 1000<=charged<=10000 or charged%1000:
        p.error('Charged budget must be 1000,2000,...,10000')
    rounds=charged//1000-1
    seconds=args.max_seconds if args.max_seconds is not None else (120 if charged==10000 else 3)
    if seconds<=0:
        p.error('Positive solver time required')
    rows,config,evidence=verify_partial(args.history,rounds,args.benchmark_repo,args.target,args.seed)
    old,old_config,old_evidence=verify_partial(args.previous,rounds,args.benchmark_repo,args.target,args.seed)
    if old_config['grid_sha256']!=config['grid_sha256'] or old_config['warm']!=config['warm']:
        raise ValueError('Previous SynFactor grid/warm mismatch')
    warm,_=warm_evidence(args.benchmark_repo,args.target,args.seed)
    cells=[('synfactor_v2',rows,True,evidence),('synfactor_v1',old,True,old_evidence)]
    for ref,source,payload,_ in baseline_cells(args.orion,args.target,args.seed):
        selected,proof=baseline_prefix(ref,payload['rows'],rounds,args.benchmark_repo,config,warm)
        proof.update(archive_sha256=sha(source),archive_path=str(source))
        cells.append((ref['method'],selected,False,proof))
    args.out.mkdir(parents=True,exist_ok=True)
    report=dict(target=args.target,seed=args.seed,charged_calls=charged,adaptive_rounds=rounds,
                grid_sha256=config['grid_sha256'],warm_sha256=config['warm']['csv_sha256'],
                rdkit=rdBase.rdkitVersion,eligibility='all valid original molecules; no druglike filtering',
                fingerprint='Morgan radius2/2048/no chirality; strict Tc<0.30; k100',
                canonical_deduplication='best raw score, then product_id',
                failed_calls_remain_charged=True,comparison_verified=True,
                source_sha256={str(path):sha(path) for path in
                    (Path(__file__),Path(__file__).with_name('live_evaluate.py'))},
                solver_seconds=seconds if args.optimal else None,cells=[])
    flat=[]
    for method,input_rows,required,proof in cells:
        result,unique=evaluate(input_rows,charged,required,args.optimal,seconds,args.benchmark_repo)
        result.update(method=method,identity_evidence=proof)
        report['cells'].append(result)
        pd.DataFrame(unique).to_csv(args.out/f'{method}__all_valid.csv.gz',index=False)
        row=dict(method=method,charged_calls=charged,eligible_molecules=len(unique),**result['scores'])
        for portfolio in result['portfolios']:
            row.update({f"{portfolio['metric']}_{key}":portfolio.get(key) for key in
                        ('value','raw_mean','selected_count','status','feasible_lower_bound','upper_bound','gap','certified')})
        flat.append(row)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        pd.DataFrame(flat).to_csv(args.out/'comparison.csv',index=False)
        print(json.dumps(row,allow_nan=False),flush=True)
    command=shlex.join([sys.executable,'-m','experiments.docking.tyk2_evaluate',*sys.argv[1:]])
    (args.out/'command.txt').write_text(command+'\n')


if __name__=='__main__':
    main()
