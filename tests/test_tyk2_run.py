import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from rdkit import DataStructs

from experiments.docking import tyk2_run as driver


class Tyk2DriverTests(unittest.TestCase):
    def test_cached_proposal_cannot_drop_structure_binding_or_repeat_molecules(self):
        ids=[f'p{i}' for i in range(1000)]
        proposal=dict(target='tyk2',seed=0,run_id='run',batch_id='batch_01',product_ids=ids,
                      submitted_smiles={pid:f'[{i+100}CH4]' for i,pid in enumerate(ids)})
        space=SimpleNamespace(decode=Mock())
        validate=lambda value,history=[]:driver.validate_proposal(value,history,space,
            target='tyk2',seed=0,run_id='run',batch_id='batch_01')
        self.assertEqual(validate(proposal),ids)
        changed=copy.deepcopy(proposal);del changed['submitted_smiles']['p0']
        with self.assertRaisesRegex(ValueError,'bind every route'):validate(changed)
        changed=copy.deepcopy(proposal);changed['submitted_smiles']['p1']=changed['submitted_smiles']['p0']
        with self.assertRaisesRegex(ValueError,'canonical, valid and distinct'):validate(changed)
        with self.assertRaisesRegex(ValueError,'already charged molecule'):
            validate(proposal,[dict(product_id='old',smiles=proposal['submitted_smiles']['p0'])])
        changed=copy.deepcopy(proposal);changed['submitted_smiles']['p0']='C(C)'
        with self.assertRaisesRegex(ValueError,'canonical, valid and distinct'):validate(changed)
        changed=copy.deepcopy(proposal);changed['seed']=1
        with self.assertRaisesRegex(ValueError,'another experiment'):validate(changed)

    def test_prepare_round_fits_both_models_to_same_observed_labels_and_keeps_filtered_indices(self):
        history=[dict(product_id=f'w{i}',smiles=f'w{i}',score=-float(i+1),status='valid') for i in range(12)]
        history.append(dict(product_id='failed',smiles='failed',score=-999.,status='failed'))
        fitted={}
        class CP:
            def __init__(self,*args,**kwargs):
                self.model=SimpleNamespace(state_dict=lambda:{})
                self.x_mean=self.x_scale=np.zeros(1);self.y_mean=self.y_scale=1.
            def fit(self,rows,rids,y,**kwargs):fitted['cp']=y.copy();return {}
        class XGB:
            def __init__(self,*args,**kwargs):self.model=SimpleNamespace(save_model=lambda path:None)
            def fit_features(self,features,y,**kwargs):fitted['xgb']=y.copy();return {}
            def predict_features(self,features):return np.arange(len(features),0,-1,dtype=float)
        class Assembly:
            last_stats={}
            def assemble(self,ids):
                return [dict(product_id=p,smiles=p,status='failed' if p in ('p0','p17') else 'valid') for p in ids]
            def fingerprints(self,ids):
                result=[]
                for pid in ids:
                    value=DataStructs.ExplicitBitVect(2048)
                    value.SetBit(int(pid[1:])%100)
                    result.append(value)
                return result
        space=SimpleNamespace(features=np.zeros((1,1)),reactions=['r'],
                              decode=lambda ids:(np.zeros((len(ids),3),int),np.zeros(len(ids),int)))
        pool=dict(product_ids=[f'p{i}' for i in range(2005)],cp_scores=np.arange(2005,dtype=float),
                  source=['global_cp_0']*2005,diagnostics={})
        config=dict(seed=0,tail_weight=1,weight_cp=False,policy='raw',global_k=2005,
                    local_hits=0,local_per_hit=0,random_pool_count=0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(driver,'WeightedFactorRegressor',CP),patch.object(driver,'ProductXGBRegressor',XGB), \
             patch.object(driver,'build_candidate_pool',return_value=pool) as builder, \
             patch.object(driver,'canonical',side_effect=lambda x:x), \
             patch.object(driver.torch,'save'),patch.object(driver.torch.cuda,'empty_cache'):
            ids,smiles,diagnostics=driver.prepare_round(space,Assembly(),history,config,1,Path(tmp))
            portfolio_dir=Path(tmp)/'portfolio';portfolio_dir.mkdir()
            portfolio_ids,_,portfolio_diagnostics=driver.prepare_round(
                space,Assembly(),history,{**config,'policy':'portfolio'},1,portfolio_dir)
        np.testing.assert_array_equal(fitted['cp'],np.arange(12,0,-1))
        np.testing.assert_array_equal(fitted['xgb'],fitted['cp'])
        self.assertIs(builder.call_args.args[2],history)
        self.assertEqual(ids[:2],['p1','p2'])
        self.assertEqual(ids[15:17],['p16','p18'])
        self.assertEqual(diagnostics['cp_predictions'][:2],[1.,2.])
        self.assertEqual(len(ids),1000)
        self.assertEqual(diagnostics['selection']['counts'],{'exploit':900,'rank_disagreement':100})
        self.assertEqual(diagnostics['train_count'],12)
        self.assertEqual(len(portfolio_diagnostics['selection']['groups']),9)
        self.assertEqual(len(portfolio_ids),1000)
        # One-hot true-product fingerprints produce100 distinct families per
        # strict portfolio, with those same families allowed in later portfolios.
        for start in range(0,900,100):
            self.assertEqual(len({int(pid[1:])%100 for pid in portfolio_ids[start:start+100]}),100)
        self.assertEqual({int(pid[1:])%100 for pid in portfolio_ids[:100]},
                         {int(pid[1:])%100 for pid in portfolio_ids[100:200]})

    def test_fresh_two_round_loop_retrains_only_after_charged_response_and_resume_does_not_requery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'library';repo=root/'benchmark'
            args=SimpleNamespace(out=root/'run',cache=root/'cache',library=library,assembly_cache=root/'assembly',
                assembler_source=root/'native',benchmark_repo=repo,host='mock',key=root/'key',remote_python='python',
                remote_root='remote',seed=0,policy='raw',tail_weight=1,weight_cp=False,global_k=10000,
                local_hits=0,local_per_hit=0,random_pool_count=0,rounds=2,execute=True)
            warm=[dict(product_id=f'w{i}',smiles=f'w{i}',score=-3.,status='valid',iteration=-1,
                       source='charged_shared_warm',charged=True) for i in range(1000)]
            hashes=dict(ids_sha256='ids',matrix_sha256='matrix',csv_sha256='csv')
            protocol=dict(warm_ordered_ids_sha256='ids',grid_sha256='grid',source_sha256={
                str(repo/'artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv'):'matrix',
                str(library/'reactions.txt'):'reaction'})
            space=SimpleNamespace(manifest={'source_sha256':{'reactions.txt':'reaction'}},decode=Mock())
            assembly=SimpleNamespace(identity={},close=Mock())
            observed_lengths=[];queried=[]
            def prepare(space,assembly,history,config,iteration,folder):
                observed_lengths.append(len(history))
                if iteration==2:
                    self.assertEqual(sum(r['source']=='live_glide' for r in history),1000)
                    self.assertEqual(sum(r['status']=='failed' for r in history),1)
                ids=[f'b{iteration}_{i}' for i in range(1000)]
                return ids,{p:p for p in ids},dict(search_seconds=.1)
            def query(proposal,folder):
                queried.append(proposal['product_ids'])
                return dict(protocol=protocol,seconds=.2,rows=[dict(product_id=p,smiles=p,
                    score=-4. if i else None,valid=bool(i)) for i,p in enumerate(proposal['product_ids'])])
            oracle=SimpleNamespace(preflight=Mock(return_value=protocol),query=Mock(side_effect=query))
            with patch.object(driver,'FullSpace',return_value=space),patch.object(driver,'AssemblyCache',return_value=assembly), \
                 patch.object(driver,'RemoteOracle',return_value=oracle),patch.object(driver,'warm_start',return_value=(warm,hashes)), \
                 patch.object(driver,'prepare_round',side_effect=prepare),patch.object(driver,'canonical',side_effect=lambda x:x):
                driver.run(args)
                driver.run(args)
            state=json.loads((args.out/'state.json').read_text())
            self.assertEqual(observed_lengths,[1000,2000])
            self.assertEqual(len(queried),2)
            self.assertEqual(len(state['history']),3000)
            self.assertEqual(len({r['product_id'] for r in state['history']}),3000)
            self.assertEqual(state['rounds'][-1]['calls'],3000)
            self.assertFalse((args.out/'CELL_DONE').exists())


if __name__=='__main__':unittest.main()
