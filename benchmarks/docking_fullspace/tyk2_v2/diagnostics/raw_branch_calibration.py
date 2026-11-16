from pathlib import Path
import json,torch
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem
from experiments.docking.factor_model import _FactorNetwork
from experiments.docking.tyk2_models import product_morgan
p=Path('runs/tyk2_v2/cells/tyk2_raw_uniform_s0');state=json.loads((p/'state.json').read_text());search=state['rounds'][0]['search'];checkpoint=torch.load(p/'batch_01/cp.pt',map_location='cpu',weights_only=False)
train_rows=[state['history'][i] for i in search['train_history_indices']];frame=pd.read_parquet('runs/docking_fullspace/freedom/synthons.parquet',columns=['reaction_id','position','synthon_id']);index=pd.MultiIndex.from_frame(frame[['reaction_id','position','synthon_id']]);keys=[];rows=np.full((len(train_rows),3),-1,dtype=np.int64);rids=[];reaction_names=json.loads(Path('runs/docking_fullspace/freedom/manifest.json').read_text())['reactions'];ridmap={r:i for i,r in enumerate(reaction_names)}
for i,row in enumerate(train_rows):
 rid,*sids=row['product_id'].split('____');positions=index.get_indexer([(rid,j+1,sid) for j,sid in enumerate(sids)]);assert (positions>=0).all();rows[i,:len(positions)]=positions;rids.append(ridmap[rid])
used=np.unique(rows[rows>=0]);local=np.where(rows>=0,np.searchsorted(used,rows),-1);features=np.load('runs/docking_fullspace/freedom/features.npy',mmap_mode='r')[used];x=np.clip((features-checkpoint['x_mean'])/checkpoint['x_scale'],-8,8)
torch.set_num_threads(2);model=_FactorNetwork(516,len(reaction_names),64,128,True);model.load_state_dict(checkpoint['state_dict']);model.eval()
with torch.no_grad():pred=(model(torch.tensor(x),torch.tensor(local),torch.tensor(rids)).numpy()*checkpoint['y_scale']+checkpoint['y_mean'])
y=np.array([-float(r['score']) for r in train_rows]);order=np.random.default_rng(17+7919).permutation(len(y));cut=int(len(y)*.2)
for name,ix in [('train',order[cut:]),('validation',order[:cut])]:print('checkpoint',name,'nmse',float(np.mean((pred[ix]-y[ix])**2)/checkpoint['y_scale']**2),'rho',float(spearmanr(pred[ix],y[ix]).statistic),'predsd',float(np.std(pred[ix])),'ysd',float(np.std(y[ix])))
selected=pd.read_csv(p/'batch_01/selected.csv');selected['score']=pd.to_numeric(pd.DataFrame(state['history'][1000:2000]).score);bits=product_morgan(selected.smiles);packed=np.packbits(bits,axis=1);unique,inverse,counts=np.unique(packed,axis=0,return_inverse=True,return_counts=True);selected['fp_group']=inverse;selected['connectivity']=selected.smiles.map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False))
print('fingerprints',len(unique),'groupswithduplicates',int((counts>1).sum()),'rowsinduplicategroups',int(counts[counts>1].sum()),'uniqueconnectivity',selected.connectivity.nunique())
ranges=selected.groupby('fp_group').score.agg(['count','min','max']);ranges=ranges[ranges['count']>1];print('duplicategroupmeanrange',float((ranges['max']-ranges['min']).mean()),'maxrange',float((ranges['max']-ranges['min']).max()),'groups_range>1',int(((ranges['max']-ranges['min'])>1).sum()))
print('dominantreaction top100', selected.loc[selected.score.nsmallest(100).index].product_id.str.split('____').str[0].value_counts().to_dict())

import hashlib
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
calibration = {}
for group in ['candidate_source','acquisition_source']:
 calibration[group] = {}
 for name, block in selected.groupby(group):
  calibration[group][name] = dict(n=len(block),valid=len(block),mean_raw_score=float(block.score.mean()),
   best_raw_score=float(block.score.min()),new_batch_top100_count=len(set(block.product_id)&set(selected.nsmallest(100,'score').product_id)),
   cp_mean_optimism=float((block.cp+block.score).mean()),xgb_mean_optimism=float((block.xgb+block.score).mean()),
   cp_spearman=float(spearmanr(block.cp,-block.score).statistic),xgb_spearman=float(spearmanr(block.xgb,-block.score).statistic))
fp_stats = {}
for chi in [False,True]:
 generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=chi)
 matrix=np.empty((len(selected),2048),np.uint8)
 for i,smiles in enumerate(selected.smiles):DataStructs.ConvertToNumpyArray(generator.GetFingerprint(Chem.MolFromSmiles(smiles)),matrix[i])
 _,counts=np.unique(np.packbits(matrix,axis=1),axis=0,return_counts=True)
 fp_stats[str(chi)]=dict(unique_vectors=len(counts),duplicate_groups=int((counts>1).sum()),rows_in_duplicate_groups=int(counts[counts>1].sum()))
validation={}
for name,ix in [('fit',order[cut:]),('validation',order[:cut])]:
 validation[name]=dict(n=len(ix),nmse=float(np.mean((pred[ix]-y[ix])**2)/checkpoint['y_scale']**2),spearman=float(spearmanr(pred[ix],y[ix]).statistic))
validation['constant_train_mean_validation_nmse']=float(np.mean((y[order[:cut]]-checkpoint['y_mean'])**2)/checkpoint['y_scale']**2)
validation['selected_epoch']=search['cp_fit']['selected_epoch']
sourcepaths=[p/'state.json',p/'batch_01/selected.csv',p/'batch_01/cp.pt',p/'batch_01/candidate_pool.csv.gz',Path(__file__)]
result=dict(protocol='CPU-only checkpoint/prediction analysis using shared warm and acquired rawbranch batch1 only; no model refits or new labels',
 limitations=['Adaptive selection-biased source comparisons; not causal arm effects.','Fingerprint collisions suggest a possible feature limitation; chirality-aware ranking benefit is untested.'],
 calibration=calibration,cp_validation=validation,fingerprints=fp_stats,
 nonchiral_duplicate_score_ranges=dict(mean=float((ranges['max']-ranges['min']).mean()),maximum=float((ranges['max']-ranges['min']).max()),groups_over_one_unit=int(((ranges['max']-ranges['min'])>1).sum())),
 candidate_pool=search['pool'],selected_candidate_sources=search['selected_candidate_sources'],
 selected_dominant_reaction_fraction=float(selected.product_id.str.split('____').str[0].value_counts().iloc[0]/len(selected)),
 source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sourcepaths})
Path('runs/tyk2_v2/offline_models/raw_branch_calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
