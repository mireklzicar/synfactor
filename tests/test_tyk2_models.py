import unittest
import importlib.util
import numpy as np
from experiments.docking.factor_model import FactorRegressor
from experiments.docking.tyk2_models import (ProductXGBRegressor, WeightedFactorRegressor,
    product_morgan, training_labels, tail_weights, GA_V14_XGB_PARAMS)


class Tyk2ModelsTests(unittest.TestCase):
    def test_training_ledger_excludes_failures_and_sentinels_without_fake_labels(self):
        rows=[dict(status='valid',score=-5),dict(status='failed',score=-99),
              dict(status='valid',score=10000),dict(status='valid',score=None),
              dict(status='valid',score=-8)]
        ids,y=training_labels(rows)
        np.testing.assert_array_equal(ids,[4,0]);np.testing.assert_array_equal(y,[8,5])
        w=tail_weights(np.array([1,2,2,4]),maximum=4)
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertTrue(np.all(w>0));self.assertLessEqual(float(w.max()/w.min()),4)
        self.assertEqual(w[1],w[2]);self.assertGreater(w[3],w[0])

    def test_true_product_bits_include_junctions_but_not_chirality(self):
        bits=product_morgan(['CCO','C.CO','C[C@H](O)F','C[C@@H](O)F'])
        self.assertEqual(bits.shape,(4,2048));self.assertEqual(bits.dtype,np.uint8)
        self.assertFalse(np.array_equal(bits[0],bits[1]))
        np.testing.assert_array_equal(bits[2],bits[3])

    @unittest.skipUnless(importlib.util.find_spec('xgboost') and importlib.util.find_spec('sklearn'),
                         'Optional docking XGBoost/scikit-learn dependencies are not installed')
    def test_xgb_features_fit_independent_labels_with_exact_preset(self):
        x=np.zeros((40,2048),np.uint8);x[20:,3]=1;y=np.r_[np.zeros(20),np.ones(20)*4]
        model=ProductXGBRegressor(seed=0);meta=model.fit_features(x,y)
        pred=model.predict_features(x,batch_size=7)
        self.assertLess(np.mean((pred-y)**2),.01)
        for key,value in GA_V14_XGB_PARAMS.items():self.assertEqual(meta['params'][key],value)
        with self.assertRaises(ValueError):model.fit_features(x,y,sample_weight=np.zeros(40))

    def test_default_cp_is_unchanged_and_reaction_variant_is_explicit(self):
        rng=np.random.default_rng(7);features=rng.normal(size=(12,3)).astype(np.float32)
        rows=rng.integers(0,12,size=(30,3));rows[:15,2]=-1;rids=rng.integers(0,2,30);y=rng.normal(size=30)
        original=FactorRegressor(features,2,rank=4,hidden=8,device='cpu',seed=1)
        unchanged=WeightedFactorRegressor(features,2,rank=4,hidden=8,device='cpu',seed=1)
        original.fit(rows,rids,y,epochs=2);unchanged.fit(rows,rids,y,epochs=2)
        np.testing.assert_array_equal(original.predict(rows,rids),unchanged.predict(rows,rids))
        conditioned=WeightedFactorRegressor(features,2,rank=4,hidden=8,device='cpu',seed=1,
                                            synthon_reaction_ids=np.arange(12)%2)
        result=conditioned.fit(rows,rids,y,epochs=2,sample_weight=tail_weights(y))
        self.assertTrue(result['reaction_conditioning']);self.assertTrue(np.isfinite(conditioned.predict(rows,rids)).all())
        np.testing.assert_array_equal(conditioned.x[:,-1].numpy(),np.arange(12)%2)


if __name__=='__main__':unittest.main()
