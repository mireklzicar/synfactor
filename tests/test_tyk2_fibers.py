import itertools
import unittest

import numpy as np
import torch
from rdkit import DataStructs

from experiments.docking.factor_model import FactorRegressor, _FactorNetwork
from experiments.docking.tyk2_fibers import cp_fiber_candidates, observed_anchors


class Space:
    def __init__(self):
        self.features=np.zeros((12,2),np.float32)
        self.grids=[dict(reaction_index=0,slots=[np.arange(3),np.arange(3,6)]),
                    dict(reaction_index=1,slots=[np.arange(6,8),np.arange(8,10),np.arange(10,12)])]
    def ids(self,rows,rids):
        return [':'.join(map(str,(int(r),*map(int,row)))) for row,r in zip(rows,rids)]
    def decode(self,ids):
        values=np.asarray([list(map(int,p.split(':'))) for p in ids],np.int64).reshape(-1,4)
        return values[:,1:],values[:,0]


def model_for(space,additive=True):
    torch.manual_seed(31)
    model=FactorRegressor(space.features,2,rank=4,hidden=4,additive=additive,device='cpu')
    model.model=_FactorNetwork(2,2,4,4,additive)
    model.cached_factors=torch.randn(3,12,4)
    model.cached_additive=torch.randn(3,12) if additive else None
    with torch.no_grad():
        model.model.bias.fill_(1.2);model.model.reaction_bias.weight[:]=torch.tensor([[.4],[-.7]])
    model.y_scale=2.3;model.y_mean=-.8
    return model


class FiberTests(unittest.TestCase):
    def history(self):
        return [dict(product_id='0:0:3:-1',status='valid',score=-8),
                dict(product_id='1:6:8:10',status='valid',score=-7),
                dict(product_id='0:1:3:-1',status='failed',score=-999)]

    def test_mixed_arity_fiber_topk_matches_direct_cached_prediction(self):
        space=Space();history=self.history();excluded={r['product_id'] for r in history}
        for additive in (False,True):
            model=model_for(space,additive)
            result=cp_fiber_candidates(space,model,history,anchor_count=2,per_fiber=1,block_size=1)
            expected={}
            for anchor in history[:2]:
                rows,rids=space.decode([anchor['product_id']]);parent=rows[0];rid=int(rids[0])
                for slot,axis in enumerate(space.grids[rid]['slots']):
                    candidates=[]
                    for value in axis:
                        row=parent.copy();row[slot]=value
                        pid=space.ids([row],[rid])[0]
                        if pid not in excluded:
                            score=float(model.predict([row],[rid])[0]);candidates.append((score,int(value),pid))
                    if candidates:
                        score,_,pid=sorted(candidates,key=lambda x:(-x[0],x[1]))[0];expected[pid]=score
            self.assertEqual(set(result['product_ids']),set(expected))
            np.testing.assert_allclose(result['cp_scores'],[expected[p] for p in result['product_ids']],rtol=1e-6,atol=2e-6)
            parent_rows,_=space.decode(result['parent_product_ids'])
            self.assertTrue(np.all((result['tuples']!=parent_rows).sum(1)==1))
            self.assertFalse(set(result['product_ids'])&excluded)
            self.assertTrue(result['diagnostics']['complete'])

    def test_tied_membership_is_independent_of_block_size_and_capacity_is_explicit(self):
        space=Space();model=model_for(space,False);model.cached_factors.fill_(1)
        a=cp_fiber_candidates(space,model,self.history(),anchor_count=1,per_fiber=1,block_size=1)
        b=cp_fiber_candidates(space,model,self.history(),anchor_count=1,per_fiber=1,block_size=10)
        self.assertEqual(a['product_ids'],b['product_ids'])
        self.assertEqual(a['product_ids'],['0:2:3:-1','0:0:4:-1'])
        with self.assertRaisesRegex(ValueError,'capacity exceeds'):
            cp_fiber_candidates(space,model,self.history(),anchor_count=2,per_fiber=100,max_candidates=10)

    def test_anchor_diversity_uses_observed_valid_labels_only(self):
        history=[dict(product_id=str(i),status='valid',score=-10+i) for i in range(5)]
        history[4].update(status='failed',score=-999)
        fps=[]
        for bit in (0,0,1,2,3):
            value=DataStructs.ExplicitBitVect(32);value.SetBit(bit);fps.append(value)
        np.testing.assert_array_equal(observed_anchors(history,policy='raw',count=3),[0,1,2])
        np.testing.assert_array_equal(observed_anchors(history,policy='gpdiv',count=3,fingerprints=fps),[0,2,3])

    def test_timeout_does_not_return_partial_fibers_and_restores_precision(self):
        space=Space();model=model_for(space)
        previous=torch.backends.cuda.matmul.fp32_precision
        with self.assertRaisesRegex(TimeoutError,'Incomplete CP fiber scan'):
            cp_fiber_candidates(space,model,self.history(),anchor_count=2,max_seconds=1e-12)
        self.assertEqual(torch.backends.cuda.matmul.fp32_precision,previous)


if __name__=='__main__':unittest.main()
