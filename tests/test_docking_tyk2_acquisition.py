import itertools
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import DataStructs

from experiments.docking.tyk2_acquisition import build_candidate_pool, observed_neighbors, select_batch


class TinySpace:
    def __init__(self):
        self.features=np.eye(10,512,dtype=np.float32)
        self.grids=[dict(reaction_index=0,slots=[np.array([0,1]),np.array([2,3])]),
                    dict(reaction_index=1,slots=[np.array([4,5]),np.array([6,7]),np.array([8,9])])]
        self.all=[(rid,*row,*([-1] if rid==0 else [])) for rid,g in enumerate(self.grids)
                  for row in itertools.product(*g['slots'])]
        self.manifest={'products':len(self.all)}
    def ids(self,rows,rids):
        return [':'.join(map(str,(int(r),*map(int,row)))) for row,r in zip(rows,rids)]
    def decode(self,ids):
        values=np.asarray([list(map(int,p.split(':'))) for p in ids],np.int64).reshape(-1,4)
        return values[:,1:],values[:,0]
    def sample(self,count,rng,excluded):
        available=[x for x in self.all if x not in excluded]
        selected=np.asarray([available[i] for i in rng.choice(len(available),count,replace=False)],np.int64)
        return selected[:,1:],selected[:,0]


def fp(bits):
    value=DataStructs.ExplicitBitVect(32)
    for bit in bits:value.SetBit(bit)
    return value


class AcquisitionTests(unittest.TestCase):
    def test_neighbors_only_use_valid_observed_hits_and_change_one_compatible_slot(self):
        space=TinySpace()
        history=[dict(product_id='0:0:2:-1',status='valid',score=-5),
                 dict(product_id='1:4:6:8',status='failed',score=-999),
                 dict(product_id='1:5:7:9',status='valid',score=-999)]
        result=observed_neighbors(space,history,hits=100,per_hit=10,seed=3)
        self.assertEqual(set(result['reaction_ids']),{0})
        self.assertEqual(set(result['parent_product_ids']),{'0:0:2:-1'})
        self.assertTrue(np.all(np.sum(result['tuples']!=[0,2,-1],axis=1)==1))
        self.assertEqual(len(result['tuples']),2)
        zero=observed_neighbors(space,history,hits=0)
        self.assertEqual(zero['tuples'].shape,(0,3))

    def test_global_pool_excludes_history_and_does_not_force_reaction_representation(self):
        space=TinySpace();history=[dict(product_id='0:0:2:-1',status='valid',score=-5)]
        class Model:
            def predict(self,rows,rids):return rows[:,:2].sum(1).astype(float)
        sweep=dict(tuples=np.array([[0,3,-1],[1,2,-1]]),reaction_ids=np.array([0,0]),
                   scores=np.array([3.,3.]),diagnostics={'visited_products':12})
        with patch('experiments.docking.tyk2_acquisition.stream_topk',return_value=sweep) as scorer:
            result=build_candidate_pool(space,Model(),history,global_k=2,local_hits=0,random_pool_count=0)
        self.assertEqual(result['product_ids'],['0:0:3:-1','0:1:2:-1'])
        self.assertEqual(result['diagnostics']['reaction_count'],1)
        np.testing.assert_array_equal(scorer.call_args.kwargs['excluded_tuples'],[[0,2,-1]])
        np.testing.assert_array_equal(result['cp_scores'],[3.,3.])

    def test_multimodel_union_divides_global_capacity_and_rescores_all_candidates(self):
        space=TinySpace();history=[dict(product_id='0:0:2:-1',status='valid',score=-5)]
        class Model:
            def __init__(self,offset):self.offset=offset
            def predict(self,rows,rids):return rows[:,:2].sum(1).astype(float)+self.offset
        sweeps=[dict(tuples=np.array([[0,3,-1],[1,2,-1]]),reaction_ids=np.array([0,0]),diagnostics={}),
                dict(tuples=np.array([[4,7,9]]),reaction_ids=np.array([1]),diagnostics={})]
        with patch('experiments.docking.tyk2_acquisition.stream_topk',side_effect=sweeps) as scorer:
            result=build_candidate_pool(space,[Model(0),Model(2)],history,
                global_k=3,local_hits=0,random_pool_count=0)
        self.assertEqual([c.kwargs['k'] for c in scorer.call_args_list],[2,1])
        self.assertEqual(result['cp_predictions'].shape,(3,2))
        np.testing.assert_array_equal(result['cp_scores'],[4.,4.,12.])

    def test_raw_ranking_allows_close_analogues_and_true_product_gpdiv_excludes_them(self):
        ids=['a','b','c'];fps=[fp([0,1]),fp([0,1]),fp([8])]
        args=dict(n_exploit=2,n_disagreement=0,n_random=0,product_fingerprints=fps)
        self.assertEqual(select_batch(ids,[9,8,7],mode='raw',**args)['product_ids'],['a','b'])
        self.assertEqual(select_batch(ids,[9,8,7],mode='gpdiv',**args)['product_ids'],['a','c'])
        self.assertEqual(select_batch(ids,[9,8,7],mode='novelty',novelty_weight=1,**args)['product_ids'],['a','c'])

    def test_gpdiv_boundary_and_shortfall_are_explicit(self):
        fps=[fp(range(6)),fp([0,1,2,6,7,8,9])]
        self.assertEqual(DataStructs.TanimotoSimilarity(*fps),.3)
        kwargs=dict(n_exploit=2,n_disagreement=0,n_random=0,product_fingerprints=fps,mode='gpdiv')
        with self.assertRaisesRegex(ValueError,'Only 1/2'):
            select_batch(['a','b'],[9,8],**kwargs)
        result=select_batch(['a','b'],[9,8],gpdiv_fallback='ranked',**kwargs)
        self.assertEqual(result['diagnostics']['gpdiv_ranked_fallback'],1)
        self.assertEqual(result['source'],['exploit','exploit_ranked_fallback'])

    def test_bounded_disagreement_cannot_chase_low_quality_cp_extreme(self):
        result=select_batch(list('abcdef'),[9,8,7,6,5,4],cp_scores=[8,0,7,6,5,1e30],
                            n_exploit=1,n_disagreement=1,n_random=1,seed=5)
        self.assertEqual(result['product_ids'][:2],['a','b'])
        self.assertEqual(result['source'],['exploit','rank_disagreement','random'])
        self.assertEqual(len(set(result['indices'])),3)

    def test_canonical_duplicates_failed_assembly_and_observed_ids_are_not_requeried(self):
        result=select_batch(list('abcdef'),[9,8,7,6,5,4],n_exploit=2,n_disagreement=0,n_random=0,
            molecule_keys=['same','same','failed','seen','other','last'],excluded_ids=['d'],
            product_fingerprints=[fp([1]),fp([1]),None,fp([2]),fp([3]),fp([4])])
        self.assertEqual(result['product_ids'],['a','e'])


if __name__=='__main__':unittest.main()
