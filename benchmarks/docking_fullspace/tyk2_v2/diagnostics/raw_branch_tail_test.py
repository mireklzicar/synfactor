"""Warm-only XGB weighting diagnostic on an already acquired adaptive batch."""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from experiments.docking.tyk2_models import ProductXGBRegressor,product_morgan,training_labels,tail_weights
p=Path('runs/tyk2_v2/cells/tyk2_raw_uniform_s0');out=Path('runs/tyk2_v2/offline_models');state=json.loads((p/'state.json').read_text());warm=state['history'][:1000];train_ids,y=training_labels(warm)
selected=pd.read_csv(p/'batch_01/selected.csv');observed=pd.DataFrame(state['history'][1000:2000]);assert selected.product_id.tolist()==observed.product_id.tolist()
selected['score']=pd.to_numeric(observed.score);selected['status']=observed.status
# Product IDs, not known test scores, break prediction ties.
test=selected[(selected.status=='valid')&np.isfinite(selected.score)].sort_values('product_id',kind='stable').reset_index(drop=True)
assert not set(test.product_id).intersection(warm[i]['product_id'] for i in train_ids)
train_features=product_morgan([warm[i]['smiles'] for i in train_ids]);test_features=product_morgan(test.smiles);actual=-test.score.to_numpy();results={};predictions={}
for name,weights in [('uniform',None),('tail2',tail_weights(y,maximum=2.))]:
 model=ProductXGBRegressor(seed=0,n_jobs=4);fit=model.fit_features(train_features,y,sample_weight=weights);pred=model.predict_features(test_features);predictions[name]=pred
 order=np.argsort(-pred,kind='stable');oracle=np.argsort(-actual,kind='stable');top=order[:100]
 results[name]={'fit_seconds':fit['fit_seconds'],'rmse':float(np.sqrt(np.mean((pred-actual)**2))),'spearman':float(spearmanr(pred,actual).statistic),'mean_prediction_optimism':float((pred-actual).mean()),'reranked_top100_raw_mean':float(-actual[top].mean()),'recall_of_realized_best100':len(set(top)&set(oracle[:100]))/100,'top100_source_counts':test.iloc[top].candidate_source.value_counts().to_dict()}
 if name=='uniform':results[name]['max_abs_error_against_frozen_live_prediction']=float(np.max(np.abs(pred-test.xgb.to_numpy())))
 results[name]['by_source']={}
 for source,group in test.groupby('candidate_source'):
  ix=group.index.to_numpy();truth=actual[ix];pr=pred[ix];subtop=np.argsort(-pr,kind='stable')[:min(100,len(ix))]
  results[name]['by_source'][source]={'n':len(ix),'rmse':float(np.sqrt(np.mean((pr-truth)**2))),'spearman':float(spearmanr(pr,truth).statistic),'reranked_top100_raw_mean':float(-truth[subtop].mean())}
protocol={'training':'Only997 valid shared-warm labels; no acquired-rawbatch labels used in either fit or weights','test':'All1000 newly acquired valid observations from frozen tyk2_raw_uniform_s0 batch1. Adaptive selection-biased retrospective diagnostic, not independent validation.','parameter_changes':'Only sample weights: uniform vs acquired-warm rank-tail weights, maximum ratio2, fourth power, mean normalized1','tie_policy':'prediction descending then product_id','tree_hyperparameters':'Same pinned GA-v14 XGB preset,256 trees/depth8/eta.05/subsample.8/colsample.8; seed0;CPU4threads','druglike_filter':False,'new_oracle_queries':0}
a=set(np.argsort(-predictions['uniform'],kind='stable')[:100]);b=set(np.argsort(-predictions['tail2'],kind='stable')[:100]);result={'protocol':protocol,'models':results,'reranked_top100_overlap':len(a&b),'realized_best100_raw_mean':float(-np.sort(actual)[-100:].mean()),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'state.json',p/'batch_01/selected.csv',Path('experiments/docking/tyk2_models.py'),Path(__file__)]}}
(out/'raw_branch_tail_test.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
