"""Conditional development diagnostic: fixed pre-batch3 training, chiral vs achiral XGB.

Only this raw branch's first 5000 charged rows are used. No live code changes,
new queries, hyperparameter search, or test-label use in fitting.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from experiments.docking.tyk2_models import ProductXGBRegressor, product_morgan, training_labels

CELL=Path('runs/tyk2_v2/cells/tyk2_raw_uniform_s0')
OUT=Path('runs/tyk2_v2/offline_models')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint(smiles, chirality):
 mols=[Chem.MolFromSmiles(s) for s in smiles]
 assert all(m is not None for m in mols)
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=chirality)
 bits=np.empty((len(mols),2048),np.uint8)
 for i,fp in enumerate(gen.GetFingerprints(mols,numThreads=4)):
  DataStructs.ConvertToNumpyArray(fp,bits[i])
 return bits

def metrics(frame,pred):
 actual=-frame.score.to_numpy(float)
 order=np.argsort(-pred,kind='stable');best=np.argsort(-actual,kind='stable')
 rho=float(spearmanr(pred,actual).statistic)
 result=dict(n=len(frame),rmse=float(np.sqrt(np.mean((pred-actual)**2))),spearman=rho if np.isfinite(rho) else None,mean_optimism=float((pred-actual).mean()))
 for k in (10,100):
  n=min(k,len(frame));selected=order[:n]
  result[f'predicted_top{k}_actual_raw_mean']=float(-actual[selected].mean())
  result[f'realized_top{k}_recall']=len(set(selected)&set(best[:n]))/n
  result[f'realized_best{k}_raw_mean']=float(-actual[best[:n]].mean())
 return result

def main():
 source=CELL/'state.json'; state_bytes=source.read_bytes();state=json.loads(state_bytes)
 history=state['history'][:5000]
 assert len(history)==5000
 for start,iteration in ((0,-1),(1000,1),(2000,2),(3000,3),(4000,4)):
  assert all(r['iteration']==iteration for r in history[start:start+1000])
 ledger=history[:3000];indices,y=training_labels(ledger);train=[ledger[i] for i in indices]
 tests={};sources=[Path('experiments/docking/tyk2_models.py'),Path(__file__)]
 for batch in (3,4):
  file=CELL/f'batch_{batch:02d}'/'selected.csv';sources.append(file)
  frame=pd.read_csv(file);rows=history[batch*1000:(batch+1)*1000]
  assert frame.product_id.tolist()==[r['product_id'] for r in rows]
  frame['score']=[r['score'] for r in rows];frame['status']=[r['status'] for r in rows]
  frame['score']=pd.to_numeric(frame.score,errors='coerce')
  frame=frame[(frame.status=='valid')&np.isfinite(frame.score)&(frame.score.abs()<100)].sort_values('product_id',kind='stable').reset_index(drop=True)
  assert not set(frame.product_id).intersection(r['product_id'] for r in train)
  tests[f'batch_{batch}']=frame
 protocol=dict(training='Only valid rows in charged prefix 0:3000: warm+batch1+batch2',training_labels=len(y),
  test='Batch3 and batch4 separately; both fully excluded from fitting, uniform fixed models',
  parameters='Exact GA-v14 XGB preset, CPU4 threads, seed0; only includeChirality changes',
  features='Original product SMILES, Morgan radius2/2048 bits; no standardization, no property filter',
  labels='Observed higher-better utility=-raw docking score, valid finite |score|<100',
  tie_order='Product ID sorted before stable predicted-rank selection',
  new_oracle_queries=0,future_rows_used=0,rdkit_version=rdBase.rdkitVersion,
  caveats=['Conditional development diagnostic on acquired pools selected by the nonchiral live campaign.',
    'No counterfactual acquisition gain or out-of-pool generalization is established.',
    'Batch4 live predictions used more training labels; frozen-live equality applies only to batch3.'])
 (OUT/'chiral_feature_test_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 results={};predictions={};feature_counts={}
 for name,chiral in [('nonchiral',False),('chiral',True)]:
  train_bits=fingerprint([r['smiles'] for r in train],chiral)
  if not chiral:assert np.array_equal(train_bits,product_morgan([r['smiles'] for r in train]))
  model=ProductXGBRegressor(seed=0,n_jobs=4);fit=model.fit_features(train_bits,y)
  results[name]={'fit':fit,'tests':{}};predictions[name]={}
  feature_counts[name]={'train_unique_vectors':int(np.unique(np.packbits(train_bits,axis=1),axis=0).shape[0])}
  for label,frame in tests.items():
   bits=fingerprint(frame.smiles,chiral);pred=model.predict_features(bits);predictions[name][label]=pred
   feature_counts[name][label+'_unique_vectors']=int(np.unique(np.packbits(bits,axis=1),axis=0).shape[0])
   result=metrics(frame,pred)
   for column in ('candidate_source','acquisition_source'):
    result['by_'+column]={str(s):metrics(group,pred[group.index.to_numpy()]) for s,group in frame.groupby(column)}
   if label=='batch_3' and not chiral: result['max_abs_difference_from_frozen_live_predictions']=float(np.max(np.abs(pred-frame.xgb.to_numpy())))
   results[name]['tests'][label]=result
 deltas={}
 for label in tests:
  a=results['nonchiral']['tests'][label];b=results['chiral']['tests'][label]
  deltas[label]={'rmse_improvement':a['rmse']-b['rmse'],'spearman_improvement':b['spearman']-a['spearman'],
   'top100_raw_improvement':a['predicted_top100_actual_raw_mean']-b['predicted_top100_actual_raw_mean'],
   'top100_recall_improvement':b['realized_top100_recall']-a['realized_top100_recall'],
   'predicted_top100_overlap':len(set(np.argsort(-predictions['nonchiral'][label],kind='stable')[:100])&set(np.argsort(-predictions['chiral'][label],kind='stable')[:100]))}
 result=dict(protocol=protocol,models=results,feature_counts=feature_counts,chiral_deltas_positive_better=deltas,
  used_history_prefix_sha256=hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  source_sha256={str(source):hashlib.sha256(state_bytes).hexdigest(),**{str(p):digest(p) for p in sources}})
 (OUT/'chiral_feature_test.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
