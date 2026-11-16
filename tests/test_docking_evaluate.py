"""Budget/identity invariants for post-hoc docking evaluation."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

if importlib.util.find_spec("pyarrow") is None:
    raise unittest.SkipTest("Optional docking tests require pyarrow")

from experiments.docking.evaluate import molecular_archive, read_observed, score_metrics, conflict_graphs


class DockingEvaluationTests(unittest.TestCase):
    def test_archive_dedup_preserves_budget_and_first_graph(self):
        table = pd.DataFrame({"product_id": ["a", "b", "c", "d"],
                              "smiles": ["CCO", "OCC", "CCN", "invalid"],
                              "x_glide_raw": [-4., -8., np.nan, -10.],
                              "x_glide_le": [1., 2., 0., 3.], "x_valid": [True, True, False, True]})
        queried, metrics = score_metrics(table, ["a", "b", "c", "d"], "x_glide_raw", "x", 4)
        archive, audit = molecular_archive(queried, "x_glide_raw", "x")
        self.assertEqual(archive.product_id.tolist(), ["a"])
        self.assertEqual(archive.utility.tolist(), [4.])
        self.assertEqual(audit["duplicate_molecules_excluded"], 1)
        self.assertEqual(audit["invalid_smiles_excluded"], 1)
        self.assertEqual(metrics["nonfinite_raw_scores"], 1)

    def test_observed_formats_and_charging(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "archive.json"
            p.write_text(json.dumps({"evaluated_product_ids": ["a", "a", "b"], "submitted_calls": 3}))
            ids, audit = read_observed(p)
            self.assertEqual(ids, ["a", "b"])
            self.assertEqual(audit["submitted_calls"], 3)
            p = Path(td) / "queried.csv.gz"
            pd.DataFrame({"product_id": ["a", "b"]}).to_csv(p, index=False)
            self.assertEqual(read_observed(p)[0], ["a", "b"])

    def test_similarity_boundary_and_diagonal(self):
        graphs = conflict_graphs(["CCO", "OCC", "c1ccccc1"], [0., 1.])
        self.assertTrue(graphs[0.][0, 1])
        self.assertFalse(graphs[1.].any())
        self.assertFalse(np.diag(graphs[0.]).any())
        np.testing.assert_array_equal(graphs[0.], graphs[0.].T)


if __name__ == "__main__":
    unittest.main()
