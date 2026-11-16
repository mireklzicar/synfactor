"""CPU-only conditional label ablation, raw TYK2 branch batch1 -> batch2.

No batch2 labels enter fitting; batch3 and later rows are never used. The
observed test batch was selected using the full-label model. This is not a
counterfactual acquisition campaign or an equal-budget method comparison.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from experiments.docking.tyk2_models import ProductXGBRegressor,product_morgan,training_labels

CELL=Path('runs/tyk2_v2/cells/tyk2_raw_uniform_s0')
OUT=Path('runs/tyk2_v2/offline_models')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(frame,pred):
 actual=-frame.score.to_numpy(float);order=np.argsort(-pred,kind='stable');best=np.argsort(-actual,kind='stable')
 result=dict(n=len(frame),rmse=float(np.sqrt(np.mean((pred-actual)**2))),spearman=float(spearmanr(pred,actual).statistic),mean_optimism=float((pred-actual).mean()))
 for k in (10,100):
  n=min(k,len(frame));selected=order[:n]
  result[f'predicted_top{k}_actual_raw_mean']=float(-actual[selected].mean())
  result[f'realized_top{k}_recall']=len(set(selected)&set(best[:n]))/n
  result[f'realized_best{k}_raw_mean']=float(-actual[best[:n]].mean())
 return result

def main():
 state=json.loads((CELL/'state.json').read_text())
 history=state['history'][:3000]  # Explicitly ignore available batch3/future rows.
 assert len(history)==3000
 assert all(r['iteration']==-1 for r in history[:1000])
 assert all(r['iteration']==1 for r in history[1000:2000])
 assert all(r['iteration']==2 for r in history[2000:3000])
 first=pd.read_csv(CELL/'batch_01/selected.csv');second=pd.read_csv(CELL/'batch_02/selected.csv')
 assert first.product_id.tolist()==[r['product_id'] for r in history[1000:2000]]
 assert second.product_id.tolist()==[r['product_id'] for r in history[2000:3000]]
 omitted=set(first.loc[first.acquisition_source=='rank_disagreement','product_id']);assert len(omitted)==100
 ledger=history[:2000];train_indices,y=training_labels(ledger)
 records=[ledger[i] for i in train_indices];keep=np.array([r['product_id'] not in omitted for r in records])
 assert len(y)==1997 and int(keep.sum())==1897
 # Original observed molecular forms, with no property filter or standardization.
 train_bits=product_morgan([r['smiles'] for r in records])
 test=second.copy();test['score']=[r['score'] for r in history[2000:3000]];test['status']=[r['status'] for r in history[2000:3000]]
 test['score']=pd.to_numeric(test.score,errors='coerce')
 test=test[(test.status=='valid')&np.isfinite(test.score)&(test.score.abs()<100)].sort_values('product_id',kind='stable').reset_index(drop=True)
 assert not set(test.product_id).intersection(r['product_id'] for r in records)
 test_bits=product_morgan(test.smiles)
 protocol=dict(training_branches=['warm+batch1 all valid','warm+batch1 exploitation only; omit100 disagreement labels'],training_counts=[1997,1897],
   labels='Direct observed utility=-raw docking score; valid finite |score|<100; no distillation or property gate',
   fit='Fixed GA-v14 XGB preset, uniform sample weights, seed0, CPU4 threads; no early stopping or holdout tuning',
   test='Already observed raw branch batch2 only; score-blind product-ID tie order',future_rows_used=0,new_oracle_queries=0,
   caveats=['Different training counts: additional100 labels may help through data quantity, not disagreement-specific information.',
            'Batch2 selection depended on the all-label model, so these conditional retrospective metrics do not establish causal acquisition benefit.',
            'No counterfactual campaign, budget-matched win, or result on unqueried molecules is claimed.'])
 # Persist design before fitting either ablation model.
 (OUT/'disagreement_label_ablation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 results={};predictions={}
 for name,mask in [('all_valid',np.ones(len(y),bool)),('omit_disagreement',keep)]:
  model=ProductXGBRegressor(seed=0,n_jobs=4);fit=model.fit_features(train_bits[mask],y[mask]);pred=model.predict_features(test_bits);predictions[name]=pred
  result=metrics(test,pred);result.update(training_labels=int(mask.sum()),fit_seconds=fit['fit_seconds'],params=fit['params'])
  result['by_candidate_source']={source:metrics(group,pred[group.index.to_numpy()]) for source,group in test.groupby('candidate_source')}
  result['by_acquisition_source']={source:metrics(group,pred[group.index.to_numpy()]) for source,group in test.groupby('acquisition_source')}
  if name=='all_valid':result['max_abs_difference_from_frozen_live_predictions']=float(np.max(np.abs(pred-test.xgb.to_numpy())))
  results[name]=result
 a,b=results['all_valid'],results['omit_disagreement']
 deltas={'rmse_improvement_from_100_disagreement_labels':b['rmse']-a['rmse'],'spearman_improvement_from_100_disagreement_labels':a['spearman']-b['spearman'],
         'top100_raw_score_improvement_from_100_disagreement_labels':b['predicted_top100_actual_raw_mean']-a['predicted_top100_actual_raw_mean'],
         'top100_recall_improvement_from_100_disagreement_labels':a['realized_top100_recall']-b['realized_top100_recall']}
 result={'protocol':protocol,'models':results,'conditional_deltas_positive_means_all_labels_better':deltas,
   'top100_predicted_set_overlap':len(set(np.argsort(-predictions['all_valid'],kind='stable')[:100])&set(np.argsort(-predictions['omit_disagreement'],kind='stable')[:100])),
   'used_history_prefix_sha256':hashlib.sha256(json.dumps(history,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
   'source_sha256':{str(p):digest(p) for p in [CELL/'state.json',CELL/'batch_01/selected.csv',CELL/'batch_02/selected.csv',Path('experiments/docking/tyk2_models.py'),Path(__file__)]}}
 (OUT/'disagreement_label_ablation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
