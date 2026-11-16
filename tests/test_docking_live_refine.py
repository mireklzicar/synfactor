import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

AVAILABLE = importlib.util.find_spec('pyscipopt') is not None and importlib.util.find_spec('scipy') is not None
BENCHMARK = Path('/home/mirekl/SynthonBench_ICML/synthonbench_v2')
if AVAILABLE:
    from experiments.docking.live_refine import refine


@unittest.skipUnless(AVAILABLE and (BENCHMARK/'scripts/analyze_diversity_portfolios.py').exists(),
                     'Optional refinement requires pyscipopt, scipy and original Orion sources')
class RefinementTests(unittest.TestCase):
    def fixture(self, root):
        path=root/'molecules.csv'
        pd.DataFrame([dict(product_id=f'p{i:03d}',smiles='C',score=-200+i)
                      for i in range(101)]).to_csv(path,index=False)
        return path

    def test_exact_fixed_cardinality_conflict_solution_uses_mean_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(Path(tmp))
            graph=np.zeros((101,101),dtype=bool);graph[0,1]=graph[1,0]=True
            with patch('experiments.docking.live_refine.strict_conflicts',return_value=graph):
                result=refine(path,seconds=2,lp_seconds=2,support_seconds=2)
            expected=-np.delete(np.arange(-200.,-99.),1).mean()
            self.assertTrue(result['certified'])
            self.assertEqual(len(result['selected_product_ids']),100)
            self.assertNotIn('p001',result['selected_product_ids'])
            self.assertAlmostEqual(result['value'],expected)
            self.assertAlmostEqual(result['upper_bound'],expected)
            self.assertGreaterEqual(result['upper_bound'],result['value'])

    def test_support_certificate_never_becomes_full_domain_bound(self):
        sys.path.insert(0,str(BENCHMARK/'scripts'))
        source=importlib.import_module('analyze_diversity_portfolios')
        refinement=importlib.import_module('refine_diversity_bounds')
        full=dict(selected=list(range(1,101)),value=-149.5,bound=-151.,gap=1.5,
                  certified=False,status='timelimit')
        # A restricted support can certify its own optimum, but this is not a
        # certificate for all101 input candidates.
        support=dict(selected=list(range(100)),value=-150.5,bound=-150.5,gap=0.,
                     certified=True,status='optimal')
        lp=dict(bound=-150.8,selected_support=list(range(100)),rounding_order=list(range(101)))
        with tempfile.TemporaryDirectory() as tmp:
            path=self.fixture(Path(tmp))
            with patch('experiments.docking.live_refine.strict_conflicts',return_value=np.zeros((101,101),dtype=bool)), \
                 patch.object(source,'solve_independent',side_effect=[full,support]), \
                 patch.object(refinement,'continuous_bound',return_value=lp):
                result=refine(path,seconds=2,lp_seconds=2,support_seconds=2)
            self.assertEqual(result['value'],150.5)
            self.assertEqual(result['upper_bound'],150.8)
            self.assertAlmostEqual(result['gap'],.3)
            self.assertFalse(result['certified'])
            self.assertFalse(result['support_bound_used_globally'])


if __name__=='__main__':
    unittest.main()
