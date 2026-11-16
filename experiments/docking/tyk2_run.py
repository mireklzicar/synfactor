"""TYK2 development: independently fitted CP64 proposals and product-XGB ranking.

Only the charged history enters training or local-hit generation. Candidate
assembly is label-free. There is no teacher distillation or druglike filter.
Default invocation prepares one batch; --execute explicitly submits it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
import torch

from .fullspace import FullSpace
from .live import RemoteOracle, digest, protocol_digest, validate_state, warm_start, write
from .tyk2_assembly import AssemblyCache
from .tyk2_acquisition import build_candidate_pool, select_batch
from .tyk2_models import ProductXGBRegressor, WeightedFactorRegressor, tail_weights, training_labels


def fp_matrix(fingerprints):
    result = np.empty((len(fingerprints), 2048), np.uint8)
    for i, fp in enumerate(fingerprints):
        if fp is None:
            raise ValueError('Invalid assembled product cannot be represented by zero bits')
        DataStructs.ConvertToNumpyArray(fp, result[i])
    return result


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def validate_proposal(proposal, history, space, *, target, seed, run_id, batch_id):
    """Recheck a fresh or resumed proposal before any irreversible oracle work."""
    identity = dict(target=target, seed=seed, run_id=run_id, batch_id=batch_id)
    if any(proposal.get(k)!=v for k,v in identity.items()):
        raise ValueError('Committed proposal belongs to another experiment')
    ids = proposal.get('product_ids')
    if (not isinstance(ids,list) or len(ids)!=1000 or any(not isinstance(pid,str) for pid in ids)
            or len(set(ids))!=1000 or set(ids)&{r['product_id'] for r in history}):
        raise ValueError('Committed proposal violates the charged ledger')
    supplied = proposal.get('submitted_smiles')
    if not isinstance(supplied,dict) or set(supplied)!=set(ids):
        raise ValueError('Committed proposal must bind every route to its assembled structure')
    keys = [canonical(supplied[pid]) for pid in ids]
    if (any(key is None or key!=supplied[pid] for pid,key in zip(ids,keys))
            or len(set(keys))!=1000):
        raise ValueError('Committed proposal structures must be canonical, valid and distinct')
    observed_keys = {canonical(r.get('smiles')) for r in history}
    if set(keys)&observed_keys:
        raise ValueError('Committed proposal repeats an already charged molecule')
    space.decode(ids)
    return ids


def prepare_round(space, assembly, history, config, iteration, folder):
    started = time.perf_counter()
    train_indices, y = training_labels(history)
    train_ids = [history[i]['product_id'] for i in train_indices]
    tuples, reactions = space.decode(train_ids)
    weights = None if config['tail_weight'] == 1 else tail_weights(y, maximum=config['tail_weight'])
    cp = WeightedFactorRegressor(space.features, len(space.reactions), rank=64,
        additive=True, device='cuda', seed=config['seed']+17*iteration)
    cp_fit = cp.fit(tuples, reactions, y, epochs=180,
                   sample_weight=weights if config['weight_cp'] else None)
    train_products = assembly.assemble(train_ids)
    if any(r['status'] != 'valid' for r in train_products):
        raise ValueError('A valid observed molecule failed native assembly')
    if any(canonical(history[i]['smiles']) != r['smiles'] for i,r in zip(train_indices,train_products)):
        raise ValueError('Native assembled training structures differ from charged observations')
    xgb = ProductXGBRegressor(seed=config['seed'], n_jobs=4)
    xgb_fit = xgb.fit_features(fp_matrix(assembly.fingerprints(train_ids)), y, sample_weight=weights)
    xgb.model.save_model(folder/'xgb.ubj')
    torch.save(dict(state_dict=cp.model.state_dict(), x_mean=cp.x_mean, x_scale=cp.x_scale,
                    y_mean=cp.y_mean, y_scale=cp.y_scale), folder/'cp.pt')
    pool = build_candidate_pool(space, cp, history, global_k=config['global_k'],
        local_hits=config['local_hits'], local_per_hit=config['local_per_hit'],
        random_pool_count=config['random_pool_count'], seed=config['seed']+iteration*1009,
        max_sweep_seconds=300)
    print(f'round{iteration}: generated {len(pool["product_ids"])} routes', flush=True)
    products = assembly.assemble(pool['product_ids']); assembly_stats = dict(assembly.last_stats)
    valid = np.array([i for i,r in enumerate(products) if r['status']=='valid'], np.int64)
    if len(valid)<1000:
        raise ValueError('Too few chemically valid candidate products')
    ids = [pool['product_ids'][i] for i in valid]
    smiles = [products[i]['smiles'] for i in valid]
    fps = assembly.fingerprints(ids)
    xgb_predictions = xgb.predict_features(fp_matrix(fps))
    cp_predictions = pool['cp_scores'][valid]
    sources = [pool['source'][i] for i in valid]
    observed_keys = {canonical(r['smiles']) for r in history}
    common = dict(product_ids=ids, xgb_scores=xgb_predictions, cp_scores=cp_predictions,
        product_fingerprints=fps, molecule_keys=smiles, excluded_molecule_keys=observed_keys,
        seed=config['seed']+iteration, n_random=0)
    if config['policy']=='raw':
        result = select_batch(**common, mode='raw', n_exploit=900, n_disagreement=100)
        chosen = result['indices'].tolist(); selection_sources = result['source']
        selection_diagnostics = result['diagnostics']
    else:
        # Nine greedy portfolios share a query budget. Strict Tc<.30
        # applies within each100-member portfolio, not across the entire1000 batch.
        # This allows improving analogues of each diverse chemical series.
        chosen, selection_sources, groups = [], [], []
        for group in range(9):
            result = select_batch(**{**common,'excluded_ids':[ids[i] for i in chosen],
                'excluded_molecule_keys':observed_keys|{smiles[i] for i in chosen}},
                mode='gpdiv', n_exploit=100, n_disagreement=0, max_similarity=.30,
                gpdiv_fallback='raise')
            chosen.extend(result['indices'].tolist());selection_sources.extend([f'portfolio_{group}']*100)
            groups.append(result['diagnostics'])
        result = select_batch(**{**common,'excluded_ids':[ids[i] for i in chosen],
            'excluded_molecule_keys':observed_keys|{smiles[i] for i in chosen}},
            mode='raw', n_exploit=0, n_disagreement=100)
        chosen.extend(result['indices'].tolist());selection_sources.extend(result['source'])
        selection_diagnostics = dict(policy='nine successive GPDiv100 portfolios plus100 qualified disagreement',
                                     groups=groups, exploration=result['diagnostics'])
    chosen = np.asarray(chosen,np.int64)
    if len(chosen)!=1000 or len(set(ids[i] for i in chosen))!=1000 or len(set(smiles[i] for i in chosen))!=1000:
        raise ValueError('Selection must contain1000 new distinct route/molecule identities')
    selected_ids = [ids[i] for i in chosen]
    if set(selected_ids)&{r['product_id'] for r in history}:
        raise ValueError('Previously charged route selected')
    frame = pd.DataFrame(dict(product_id=ids,smiles=smiles,cp=cp_predictions,
        xgb=xgb_predictions,candidate_source=sources))
    frame.to_csv(folder/'candidate_pool.csv.gz',index=False)
    selected = frame.iloc[chosen].copy();selected['acquisition_source']=selection_sources
    selected.to_csv(folder/'selected.csv',index=False)
    fit_indices = train_indices.tolist()
    diagnostics = dict(cp_fit=cp_fit,xgb_fit=xgb_fit,train_count=len(y),
        charged_count=len(history),train_history_indices=fit_indices,
        pool=pool['diagnostics'],assembly=assembly_stats,selection=selection_diagnostics,
        selected_candidate_sources=dict(Counter(sources[i] for i in chosen)),
        selected_reactions=len({pid.split('____')[0] for pid in selected_ids}),
        xgb_predictions=xgb_predictions[chosen].tolist(),cp_predictions=cp_predictions[chosen].tolist(),
        search_seconds=time.perf_counter()-started,druglike_filter=False,distillation=False)
    del cp, xgb;torch.cuda.empty_cache()
    return selected_ids,{ids[i]:smiles[i] for i in chosen},diagnostics


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'cell.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        space=FullSpace(args.cache)
        assembly=AssemblyCache(args.library,args.assembly_cache,args.assembler_source)
        oracle=RemoteOracle(args.host,args.key,args.remote_python,args.remote_root)
        warm,warm_hashes=warm_start(args.benchmark_repo,'tyk2',args.seed,space)
        protocol=oracle.preflight('tyk2',args.seed)
        if protocol['warm_ordered_ids_sha256']!=warm_hashes['ids_sha256']:
            raise ValueError('Remote and local warm cohorts differ')
        matrix=args.benchmark_repo/f'artifacts/dock12_seed_screen/freedom/seed_{args.seed}/results_long.csv'
        if protocol['source_sha256'].get(str(matrix))!=warm_hashes['matrix_sha256']:
            raise ValueError('Remote and local warm-score matrices differ')
        for name,expected in space.manifest['source_sha256'].items():
            if protocol['source_sha256'][str(args.library/name)]!=expected:
                raise ValueError('Local and remote space source differs')
        code_names=['tyk2_run.py','tyk2_models.py','tyk2_acquisition.py','tyk2_assembly.py',
                    'factor_model.py','cartesian.py','fullspace.py','live.py']
        config=dict(target='tyk2',seed=args.seed,run_id=args.out.name,policy=args.policy,
            tail_weight=args.tail_weight,weight_cp=args.weight_cp,global_k=args.global_k,
            local_hits=args.local_hits,local_per_hit=args.local_per_hit,random_pool_count=args.random_pool_count,
            budget=10000,warm_charged=1000,batch_size=1000,warm=warm_hashes,
            grid_sha256=protocol['grid_sha256'],oracle_protocol_sha256=protocol_digest(protocol),
            assembly_identity=assembly.identity,druglike_filter=False,distillation=False,
            code_sha256={n:digest(Path(__file__).with_name(n)) for n in code_names})
        path=args.out/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:
            raise ValueError('Configuration changed; preserve earlier run and use a new output')
        write(path,config);write(args.out/'remote_preflight.json',protocol)
        path=args.out/'state.json'
        state=json.loads(path.read_text()) if path.exists() else dict(history=list(warm),rounds=[])
        validate_state(state,warm);write(path,state)
        for iteration in range(len(state['rounds'])+1,args.rounds+1):
            folder=args.out/f'batch_{iteration:02d}';folder.mkdir(exist_ok=True)
            proposal_path=folder/'proposal.json'
            if proposal_path.exists():
                proposal=json.loads(proposal_path.read_text());diagnostics=json.loads((folder/'search.json').read_text())
            else:
                ids,smiles,diagnostics=prepare_round(space,assembly,state['history'],config,iteration,folder)
                proposal=dict(target='tyk2',seed=args.seed,run_id=args.out.name,batch_id=folder.name,
                              product_ids=ids,submitted_smiles=smiles)
                write(folder/'search.json',diagnostics);write(proposal_path,proposal)
            ids=validate_proposal(proposal,state['history'],space,target='tyk2',seed=args.seed,
                                  run_id=args.out.name,batch_id=folder.name)
            if not args.execute:
                print('Proposal prepared; no new docking submitted',flush=True);break
            print(f'round{iteration}: submitting1000 {args.policy} choices',flush=True)
            response=oracle.query(proposal,folder)
            if protocol_digest(response['protocol'])!=config['oracle_protocol_sha256']:
                raise ValueError('Oracle protocol changed')
            state['history'].extend(dict(product_id=r['product_id'],smiles=r['smiles'],score=r['score'],
                status='valid' if r['valid'] else 'failed',iteration=iteration,source='live_glide',charged=True)
                for r in response['rows'])
            state['rounds'].append(dict(iteration=iteration,calls=len(state['history']),search=diagnostics,
                                       oracle_seconds=response['seconds']))
            validate_state(state,warm);write(path,state)
            valid=sorted(r['score'] for r in state['history'] if r['status']=='valid')
            write(args.out/'summary.json',dict(calls=len(state['history']),valid=len(valid),best=valid[0],
                route_top100_mean=float(np.mean(valid[:100])),
                search_seconds=sum(r['search']['search_seconds'] for r in state['rounds']),
                oracle_seconds=sum(r['oracle_seconds'] for r in state['rounds'])))
            print(f'calls={len(state["history"])} best={valid[0]:.3f} route_top100={np.mean(valid[:100]):.4f}',flush=True)
        if len(state['history'])==10000:(args.out/'CELL_DONE').write_text('10000 charged attempts completed\n')
        assembly.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('runs/docking_fullspace/freedom'))
    p.add_argument('--library',type=Path,default=Path('/home/mirekl/TWIST_Navigator/freedom5_skach'))
    p.add_argument('--benchmark-repo',type=Path,default=Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2'))
    p.add_argument('--assembler-source',type=Path,default=Path('/home/mirekl/DeepMedChem_Prod/synthon_assembler/src'))
    p.add_argument('--assembly-cache',type=Path,default=Path('runs/tyk2_v2/assembly.sqlite'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy',choices=['raw','portfolio'],default='raw')
    p.add_argument('--tail-weight',type=float,default=1)
    p.add_argument('--weight-cp',action='store_true')
    p.add_argument('--global-k',type=int,default=100000)
    p.add_argument('--local-hits',type=int,default=100)
    p.add_argument('--local-per-hit',type=int,default=512)
    p.add_argument('--random-pool-count',type=int,default=10000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--host',default='ubuntu@3.83.143.60')
    p.add_argument('--key',type=Path,default=Path('/home/mirekl/.ssh/prototypes-cli.pem'))
    p.add_argument('--remote-python',default='/home/ubuntu/dmc-navigator-prod-v040/.venv/bin/python')
    p.add_argument('--remote-root',default='/home/ubuntu/synfactor_dock12_20261116')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not 1<=args.rounds<=9 or not 1<=args.tail_weight<=4 or args.seed not in (0,1,2):
        p.error('Require1–9 adaptive rounds, tail weight1–4 and shared seed0/1/2')
    torch.set_num_threads(4)
    run(args)


if __name__=='__main__':main()
