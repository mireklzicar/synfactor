"""Separate post-hoc OPDiv refinement using the original Orion SCIP routines.

Consumes the evaluator's full molecular domain CSV. Never changes initial
evaluation results. Restricted LP-support solves improve only the incumbent;
their bounds are never reported as full-domain bounds.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from rdkit import rdBase
import pyscipopt

from experiments.docking.live_evaluate import BENCHMARK, sha, strict_conflicts


def refine(csv_path, benchmark_repo=BENCHMARK, seconds=120, lp_seconds=30, support_seconds=30):
    started = time.perf_counter()
    sys.path.insert(0, str(benchmark_repo/'scripts'))
    source = importlib.import_module('analyze_diversity_portfolios')
    refinement = importlib.import_module('refine_diversity_bounds')
    rows = pd.read_csv(csv_path).sort_values(['score', 'product_id']).to_dict('records')
    scores = np.asarray([r['score'] for r in rows], dtype=float)
    if len(rows) < 100 or not np.isfinite(scores).all():
        raise ValueError('Need at least100 molecules with finite scores')
    graph = strict_conflicts([r['smiles'] for r in rows])
    adj = [int.from_bytes(np.packbits(row, bitorder='little').tobytes(), 'little') for row in graph]
    active, removed = source.dominate(adj, scores)
    greedy = source.greedy(adj, range(len(rows)), k=100)
    starts = [source.project_selection(greedy, removed)] if len(greedy) == 100 else []
    for scale in [.05, .1, .2, .5]:
        for seed in range(4):
            order = np.lexsort((np.arange(len(rows)), scores+np.random.default_rng(seed).gumbel(size=len(rows))*scale))
            chosen = source.greedy(adj, order, active, k=100)
            if len(chosen) == 100:
                starts.append(chosen)
    if not starts:
        raise ValueError('No feasible 100-molecule start found')
    print(f'{csv_path.name}: n={len(rows)}, active={active.bit_count()}, cliques prepared by Orion; starting SCIP', flush=True)
    preprocess_seconds = time.perf_counter()-started
    sol = source.solve_independent(adj, scores, active, starts, seconds, k=100)
    full_domain_solver = dict(sol)
    details = {}
    if not sol['certified']:
        lp = refinement.continuous_bound(adj, scores, active, lp_seconds, k=100)
        details['lp'] = {k:v for k,v in lp.items() if k not in ('selected_support', 'rounding_order')}
        rounded = source.greedy(adj, lp['rounding_order'], active, k=100)
        support = source.mask_of(lp['selected_support']+sol['selected']+rounded)
        if lp['selected_support']:
            extra = source.solve_independent(adj, scores, support,
                [sol['selected']]+([rounded] if len(rounded)==100 else []), support_seconds, k=100)
            details['support_primal_solver'] = {k:v for k,v in extra.items() if k != 'selected'}
            sol['selected'], sol['value'] = extra['selected'], extra['value']
        if lp['bound'] is not None:
            sol['bound'] = max(sol['bound'], lp['bound'])
        if sol['bound'] > sol['value']+1e-7:
            raise ValueError('Invalid full-domain refinement bound')
        sol['bound'] = min(sol['bound'], sol['value'])
        sol.update(gap=sol['value']-sol['bound'], certified=sol['value']-sol['bound']<=1e-8,
                   status='bounded_refinement')
    selected = sol['selected']
    source.validate_selection(adj, selected, k=100)
    raw_mean = float(scores[selected].mean())
    assert abs(raw_mean-sol['value'])<1e-10
    return dict(input_path=str(csv_path.resolve()),input_sha256=sha(csv_path),
        rdkit=rdBase.rdkitVersion,pyscipopt=pyscipopt.__version__,
        source_sha256={str(p):sha(p) for p in (Path(__file__),Path(source.__file__),Path(refinement.__file__),
                                              Path(__file__).with_name('live_evaluate.py'))},
        contract='All input molecular rows; Morgan radius2/2048/no chirality; STRICT Tc<0.30; k100; mean negative energy',
        eligible_molecules=len(rows),active_after_exact_domination=active.bit_count(),
        raw_mean=raw_mean,value=-raw_mean,feasible_lower_bound=-raw_mean,upper_bound=max(-raw_mean,-sol['bound']),
        gap=sol['gap'],certified=sol['certified'],status=sol['status'],
        selected_product_ids=[rows[i]['product_id'] for i in selected],
        original_full_domain_solver={k:v for k,v in full_domain_solver.items() if k!='selected'},
        refinement=details,preprocess_seconds=preprocess_seconds,wall_seconds=time.perf_counter()-started,
        configured_solver_seconds=dict(full=seconds,lp=lp_seconds,support=support_seconds),
        support_bound_used_globally=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--molecules',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--benchmark-repo',type=Path,default=BENCHMARK)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--lp-seconds',type=float,default=30)
    parser.add_argument('--support-seconds',type=float,default=30)
    args=parser.parse_args()
    if min(args.seconds,args.lp_seconds,args.support_seconds)<=0:
        parser.error('Solver durations must be positive')
    if args.out.exists():
        parser.error('Output already exists; preserve previous optimization evidence')
    result=refine(args.molecules,args.benchmark_repo,args.seconds,args.lp_seconds,args.support_seconds)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    snapshots=args.out.parent/'sources';snapshots.mkdir(exist_ok=True)
    for source,digest in result['source_sha256'].items():
        if sha(source)!=digest:
            raise RuntimeError('Refinement source changed during execution')
        shutil.copyfile(source,snapshots/(digest+'.py'))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('value','upper_bound','certified','status','wall_seconds')}),flush=True)


if __name__=='__main__':
    main()
