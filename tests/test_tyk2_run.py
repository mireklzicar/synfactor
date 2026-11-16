import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from experiments.docking import tyk2_run as driver


class Tyk2DriverTests(unittest.TestCase):
    def test_variant_cli_defaults_and_portfolio_disagreement_guard(self):
        with patch('sys.argv',['tyk2_run','--out','unused']),patch.object(driver,'run') as run, \
             patch.object(driver.torch,'set_num_threads'):
            driver.main()
            args=run.call_args.args[0]
            self.assertEqual(args.fiber_anchors,'none');self.assertFalse(args.chiral_xgb)
            self.assertEqual(args.disagreement_count,100)
        for extra in (['--policy','portfolio','--disagreement-count','0'],['--disagreement-count','1001']):
            with patch('sys.argv',['tyk2_run','--out','unused',*extra]),patch.object(driver,'run') as run, \
                 redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                driver.main()
            run.assert_not_called()

    def test_fiber_union_preserves_alignment_and_removes_observed_and_duplicate_routes(self):
        pool=dict(product_ids=['old','base','overlap'],tuples=np.arange(9).reshape(3,3),
            reaction_ids=np.array([0,1,2]),cp_scores=np.array([0.,1.,2.]),
            cp_predictions=np.array([[0.],[1.],[2.]]),source=['global']*3,
            parent_product_ids=[None]*3,diagnostics={'seconds':2.})
        fibers=dict(product_ids=['overlap','new','new'],tuples=np.arange(9,18).reshape(3,3),
            reaction_ids=np.array([3,4,5]),cp_scores=np.array([10.,11.,12.]),
            source=['cp_observed_fiber']*3,parent_product_ids=['old']*3,
            variable_slots=np.array([0,2,1]),diagnostics={'seconds':1.})
        merged=driver.append_fiber_pool(pool,fibers,[{'product_id':'old'}])
        self.assertEqual(merged['product_ids'],['base','overlap','new'])
        np.testing.assert_array_equal(merged['cp_scores'],[1.,2.,11.])
        np.testing.assert_array_equal(merged['cp_predictions'],[[1.],[2.],[11.]])
        np.testing.assert_array_equal(merged['tuples'][-1],[12,13,14])
        np.testing.assert_array_equal(merged['reaction_ids'],[1,2,4])
        np.testing.assert_array_equal(merged['variable_slots'],[-1,-1,2])
        self.assertEqual(merged['parent_product_ids'],[None,None,'old'])
        self.assertEqual(merged['diagnostics']['fiber_added'],1)

    def test_chiral_features_distinguish_enantiomers_without_changing_nonchiral_contract(self):
        smiles=['F[C@](Cl)(Br)I','F[C@@](Cl)(Br)I']
        chiral=driver.chiral_product_matrix(smiles)
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
        nonchiral=driver.fp_matrix([generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles])
        self.assertFalse(np.array_equal(chiral[0],chiral[1]))
        np.testing.assert_array_equal(nonchiral[0],nonchiral[1])

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
            def fit_features(self,features,y,**kwargs):
                fitted['xgb']=y.copy();fitted.setdefault('fit_features',[]).append(features.copy());return {}
            def predict_features(self,features):
                fitted.setdefault('prediction_features',[]).append(features.copy())
                return np.arange(len(features),0,-1,dtype=float)
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
                  source=['global_cp_0']*2005,diagnostics={},tuples=np.zeros((2005,3),int),
                  reaction_ids=np.zeros(2005,int))
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
            variant_dir=Path(tmp)/'variant';variant_dir.mkdir()
            fibers=dict(product_ids=['p2005','p1','w0'],tuples=np.zeros((3,3),int),
                reaction_ids=np.zeros(3,int),cp_scores=np.array([-1.,10000.,20000.]),
                source=['cp_observed_fiber']*3,parent_product_ids=['w1']*3,
                variable_slots=np.array([0,1,2]),diagnostics={})
            def chiral_features(smiles):
                values=np.zeros((len(smiles),2048),np.uint8);values[:,-1]=1;return values
            with patch.object(driver,'cp_fiber_candidates',return_value=fibers) as fiber_call, \
                 patch.object(driver,'chiral_product_matrix',side_effect=chiral_features) as chiral_call, \
                 patch.object(driver,'select_batch',wraps=driver.select_batch) as selector:
                variant_ids,_,variant_diagnostics=driver.prepare_round(space,Assembly(),history,
                    {**config,'fiber_anchors':'gpdiv','chiral_xgb':True,'disagreement_count':0},1,variant_dir)
                self.assertIs(fiber_call.call_args.args[2],history)
                observed_fps=fiber_call.call_args.kwargs['observed_fingerprints']
                self.assertEqual(len(observed_fps),13);self.assertIsNone(observed_fps[-1])
                self.assertEqual(list(observed_fps[0].GetOnBits()),[0])
                self.assertEqual(chiral_call.call_args_list[0].args[0],[f'w{i}' for i in range(11,-1,-1)])
                self.assertEqual(chiral_call.call_args_list[1].args[0][:2],['p1','p2'])
                self.assertTrue(all(not fp.GetBit(2047) for fp in selector.call_args.kwargs['product_fingerprints']))
                self.assertEqual(selector.call_args.kwargs['n_exploit'],1000)
                self.assertEqual(selector.call_args.kwargs['n_disagreement'],0)
                import pandas as pd
                merged=pd.read_csv(variant_dir/'candidate_pool.csv.gz').set_index('product_id')
                self.assertEqual(merged.loc['p2005','cp'],-1.)
                self.assertEqual(merged.loc['p2005','candidate_parent_product_id'],'w1')
                self.assertEqual(merged.loc['p2005','candidate_variable_slot'],0)
        np.testing.assert_array_equal(fitted['cp'],np.arange(12,0,-1))
        np.testing.assert_array_equal(fitted['xgb'],fitted['cp'])
        self.assertIs(builder.call_args.args[2],history)
        self.assertEqual(ids[:2],['p1','p2'])
        self.assertEqual(ids[15:17],['p16','p18'])
        self.assertEqual(diagnostics['cp_predictions'][:2],[1.,2.])
        self.assertEqual(len(ids),1000)
        self.assertEqual(diagnostics['selection']['counts'],{'exploit':900,'rank_disagreement':100})
        self.assertEqual(diagnostics['train_count'],12)
        self.assertFalse(diagnostics['chiral_xgb']);self.assertEqual(diagnostics['fiber_anchors'],'none')
        self.assertFalse(fitted['fit_features'][0][:,-1].any())
        self.assertTrue(fitted['fit_features'][-1][:,-1].all())
        self.assertTrue(fitted['prediction_features'][-1][:,-1].all())
        self.assertEqual(variant_diagnostics['selection']['counts'],{'exploit':1000})
        self.assertEqual(len(variant_ids),1000)
        self.assertNotIn('w0',variant_ids)
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
            config=json.loads((args.out/'config.json').read_text())
            self.assertEqual(config['fiber_anchors'],'none');self.assertFalse(config['chiral_xgb'])
            self.assertEqual(config['disagreement_count'],100)
            self.assertIn('tyk2_fibers.py',config['code_sha256'])


if __name__=='__main__':unittest.main()
