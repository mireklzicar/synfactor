import gc, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiments.docking.fullspace import FullSpace
from experiments.docking.factor_model import FactorRegressor
from experiments.docking.cartesian import stream_topk

out=Path('runs/docking_fullspace/freedom')
source=Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2')
started=time.perf_counter()
space=FullSpace(out)
assert space.manifest['products']==296350180179 and len(space.reactions)==123
warm_path=source/'artifacts/seeds/freedom_296b/seed_0/selected_10k/subselections/budget_01000_candidates.csv'
score_path=source/'artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv'
warm=pd.read_csv(warm_path,dtype={'product_id':str})
assert len(warm)==1000 and warm.product_id.nunique()==1000
scores=pd.read_csv(score_path,usecols=['product_id','target_id','valid','score'])
scores=scores[scores.target_id=='tyk2'].set_index('product_id').reindex(warm.product_id)
assert scores.target_id.notna().all()
valid=scores.valid.to_numpy(bool) & np.isfinite(scores.score.to_numpy(float))
y=np.where(valid,-scores.score.to_numpy(float),0.).astype(np.float32)
rows,rids=space.decode(warm.product_id.tolist())
assert space.ids(rows,rids)==warm.product_id.tolist()
del scores;gc.collect()
print(f'Warm rows={len(y)} valid={valid.sum()}; full products={space.manifest["products"]:,}',flush=True)
torch.cuda.reset_peak_memory_stats()
fit_start=time.perf_counter()
model=FactorRegressor(space.features,len(space.reactions),rank=64,additive=True,device='cuda',seed=17)
fit=model.fit(rows,rids,y,epochs=180)
fit_wall=time.perf_counter()-fit_start
print(f'Fit complete in {fit_wall:.3f}s; GPU peak={torch.cuda.max_memory_allocated()/2**30:.3f}GiB',flush=True)
torch.cuda.reset_peak_memory_stats()
result=stream_topk(model,space.grids,20000,excluded_tuples=rows,excluded_reaction_ids=rids,
                   block_rows=4096,block_cols=8192,max_seconds=300)
sweep_peak=torch.cuda.max_memory_allocated()/2**30
rng=np.random.default_rng(20261116)
indices=rng.choice(len(result['scores']),size=100,replace=False)
direct=model.predict(result['tuples'][indices],result['reaction_ids'][indices])
error=np.abs(direct-result['scores'][indices])
assert result['diagnostics']['complete']
assert result['diagnostics']['visited_products']==296350180179
ids=space.ids(result['tuples'],result['reaction_ids'])
assert len(set(ids))==20000 and not set(ids).intersection(warm.product_id)
np.savez_compressed(out/'first_sweep.npz',tuples=result['tuples'],reaction_ids=result['reaction_ids'],scores=result['scores'],product_ids=np.asarray(ids))
counts={space.reactions[int(rid)]:int(count) for rid,count in zip(*np.unique(result['reaction_ids'],return_counts=True))}
summary=dict(target='tyk2',seed=0,warm_calls=1000,warm_valid=int(valid.sum()),model_seed=17,
              warm_ids_sha256=hashlib.sha256('\n'.join(warm.product_id).encode()).hexdigest(),
              model='CP64 + additive; max180epochs; cuda',fit=fit,fit_wall_seconds=fit_wall,
              sweep=result['diagnostics'],peak_sweep_cuda_allocated_gib=sweep_peak,
              returned_reaction_count=len(counts),returned_by_reaction=counts,
              returned_score_max=float(result['scores'].max()),returned_score_min=float(result['scores'].min()),
              checked_prediction_count=100,max_prediction_difference=float(error.max()),
              mean_prediction_difference=float(error.mean()),gpu=torch.cuda.get_device_name(),
              total_wall_seconds=time.perf_counter()-started,
              note='Complete full Freedom factor sweep; warm labels only, no new docking or molecular assembly.')
(out/'first_sweep_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['fit','returned_by_reaction']},indent=2),flush=True)
