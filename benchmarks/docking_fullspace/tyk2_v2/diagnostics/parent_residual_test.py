"""Recorded development hypothesis: parent-residual transfer on acquired local pools.

Saved pre-query models and only prior observations determine corrections.
Batch labels are accessed only for posthoc metrics. No queries or live edits.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from experiments.docking.tyk2_models import product_morgan

CELL=Path('runs/tyk2_v2/cells/tyk2_portfolio_uniform_s0')
OUT=Path('runs/tyk2_v2/offline_models')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def valid(r):
 try:s=float(r.get('score'))
 except (ValueError,TypeError):return False
 return r.get('status')=='valid' and np.isfinite(s) and abs(s)<100

def metrics(actual,pred):
 actual=np.asarray(actual);pred=np.asarray(pred);order=np.argsort(-pred,kind='stable');best=np.argsort(-actual,kind='stable')
 rho=float(spearmanr(pred,actual).statistic)
 out=dict(n=len(actual),rmse=float(np.sqrt(np.mean((pred-actual)**2))),spearman=rho if np.isfinite(rho) else None,mean_optimism=float(np.mean(pred-actual)))
 for k in (10,100):
  n=min(k,len(actual));out[f'top{k}_actual_raw_mean']=float(-actual[order[:n]].mean());out[f'best{k}_recall']=len(set(order[:n])&set(best[:n]))/n
 return out

def fp(smiles):
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
 mols=[Chem.MolFromSmiles(s) for s in smiles];assert all(m is not None for m in mols)
 return gen.GetFingerprints(mols,numThreads=4)

def compatible(a,b):
 aa=a.split('____');bb=b.split('____')
 return len(aa)==len(bb) and aa[0]==bb[0] and sum(x!=y for x,y in zip(aa[1:],bb[1:]))==1

def main():
 source=CELL/'state.json';raw=source.read_bytes();history=json.loads(raw)['history'][:4000]
 assert len(history)==4000
 protocol=dict(branch=str(CELL),charged_prefix_used=4000,test_batches=[2,3],
  models='Saved per-batch nonchiral full-product XGB; no refits',
  parents='Prior valid top100 sorted by raw score then product ID; same reaction, exactly one differing synthon',
  parent_assignment='Highest chiral Morgan radius2/2048 Tanimoto; ties use higher observed parent utility then product ID',
  formula='corrected(candidate)=base(candidate)+Tc_chiral(candidate,parent)^3*(observed_parent_utility-base(parent))',
  formula_variants_tested=1,new_oracle_queries=0,future_rows_used=0,
  caveats=['Reconstructed compatible parent need not equal the unrecorded proposal-generating parent.',
   'Conditional development analysis of already selected local pools, not causal acquisition benefit.',
   'No test labels enter parent choice, base predictions or corrections. No property gate or standardization.'],rdkit=rdBase.rdkitVersion)
 (OUT/'parent_residual_test_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 outputs={};sources=[Path(__file__),Path('experiments/docking/tyk2_models.py')]
 for batch in (2,3):
  prior=history[:batch*1000];queried=history[batch*1000:(batch+1)*1000];assert all(r['iteration']==batch for r in queried)
  parents=sorted((r for r in prior if valid(r)),key=lambda r:(float(r['score']),r['product_id']))[:100]
  assert len(parents)==100
  folder=CELL/f'batch_{batch:02}';csv=folder/'selected.csv';model_path=folder/'xgb.ubj';sources.extend([csv,model_path])
  selected=pd.read_csv(csv);assert selected.product_id.tolist()==[r['product_id'] for r in queried]
  # Prepare all parent/candidate predictions before test-label access.
  model=XGBRegressor(n_jobs=4);model.load_model(model_path)
  parent_pred=model.predict(product_morgan([r['smiles'] for r in parents]));parent_actual=np.array([-float(r['score']) for r in parents])
  parent_fp=fp([r['smiles'] for r in parents]);local=selected[selected.candidate_source=='observed_hit_neighbor'].sort_values('product_id',kind='stable').copy()
  base=model.predict(product_morgan(local.smiles));fps=fp(local.smiles)
  corrected=base.astype(float).copy();parent_indices=[];similarities=[];match_counts=[]
  for j,(pid,fpj) in enumerate(zip(local.product_id,fps)):
   compatible_indices=[i for i,p in enumerate(parents) if compatible(pid,p['product_id'])]
   match_counts.append(len(compatible_indices))
   if not compatible_indices:parent_indices.append(-1);similarities.append(0.);continue
   tc=DataStructs.BulkTanimotoSimilarity(fpj,[parent_fp[i] for i in compatible_indices])
   best=min(range(len(compatible_indices)),key=lambda k:(-tc[k],-parent_actual[compatible_indices[k]],parents[compatible_indices[k]]['product_id']))
   i=compatible_indices[best];sim=tc[best];parent_indices.append(i);similarities.append(sim)
   corrected[j]+=sim**3*(parent_actual[i]-parent_pred[i])
  # Only now access queried docking labels, solely to compute retrospective metrics.
  truth={r['product_id']:r for r in queried};mask=np.array([valid(truth[pid]) for pid in local.product_id])
  actual=np.array([-float(truth[pid]['score']) for pid in local.product_id[mask]])
  base_valid=base[mask];corrected_valid=corrected[mask]
  parent_indices=np.array(parent_indices);similarities=np.array(similarities);matched=parent_indices>=0
  out=dict(queried_local=len(local),valid_local=int(mask.sum()),matched_local=int(matched.sum()),unmatched_local=int((~matched).sum()),
   multiple_compatible_parents=int((np.array(match_counts)>1).sum()),base=metrics(actual,base_valid),anchored=metrics(actual,corrected_valid),
   max_abs_base_difference_from_live=float(np.max(np.abs(base-local.xgb.to_numpy()))),
   mean_chiral_tc=float(similarities[matched].mean()),mean_correction=float((corrected-base)[matched].mean()),
   median_correction=float(np.median((corrected-base)[matched])),correction_range=[float((corrected-base)[matched].min()),float((corrected-base)[matched].max())],
   parent_top10={'actual_utilities':parent_actual[:10].tolist(),'predicted_utilities':parent_pred[:10].tolist(),'residuals':(parent_actual-parent_pred)[:10].tolist()},
   parent_top100={'actual_mean':float(parent_actual.mean()),'predicted_mean':float(parent_pred.mean()),'residual_mean':float((parent_actual-parent_pred).mean())})
  out['top100_predicted_overlap']=len(set(np.argsort(-base_valid,kind='stable')[:100])&set(np.argsort(-corrected_valid,kind='stable')[:100]))
  out['by_acquisition_source']={}
  for arm in sorted(local.acquisition_source.unique()):
   keep=mask&(local.acquisition_source.to_numpy()==arm)
   values=np.array([-float(truth[pid]['score']) for pid in local.product_id[keep]])
   if len(values):out['by_acquisition_source'][arm]={'base':metrics(values,base[keep]),'anchored':metrics(values,corrected[keep])}
  outputs[f'batch_{batch}']=out
 result=dict(protocol=protocol,batches=outputs,used_history_prefix_sha256=hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_sha256={str(source):hashlib.sha256(raw).hexdigest(),**{str(p):sha(p) for p in sources}})
 (OUT/'parent_residual_test.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
