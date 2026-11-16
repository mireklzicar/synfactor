"""Budget isolation, acquisition provenance and fingerprint alignment checks."""
import gzip
import inspect
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

if importlib.util.find_spec("pyarrow") is None:
    raise unittest.SkipTest("Optional docking tests require pyarrow")

from chemistry import assemble
from experiments.docking import prepare, run


def make_cache(root):
    """Two mixed-arity grids, deliberately unsorted synthon IDs and row order."""
    frame, layout, rows, reaction = [], {}, [], []
    for rid, axes in (("z", ([40, 3, 17, 11], [7, 30, 2, 18])),
                      ("a", ([9, 2], [15, 3], [8, 6, 4, 10]))):
        ss = []
        for slot, ids in enumerate(axes, 1):
            axis = []
            for sid in ids:
                axis.append(len(frame))
                frame.append(dict(reaction_id=rid, position=slot, synthon_id=sid))
            ss.append(axis)
        layout[rid] = dict(offset=len(rows), slots=ss)
        for row in itertools.product(*ss):
            rows.append(list(row) + [-1] * (3-len(row)))
            reaction.append(0 if rid == "a" else 1)
    rows = np.asarray(rows, np.int32)
    # Cache order differs from Cartesian enumeration order.
    order = np.random.default_rng(27).permutation(len(rows))
    rows, reaction = rows[order], np.asarray(reaction)[order]
    manifest = dict(layout=layout, reactions=["a", "z"], scale="1M", kind="glide",
                    bitorder="big", fingerprint_type="RDKit path 2048")
    (root / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(frame).to_csv(root / "synthons.csv", index=False)
    np.save(root / "enum_indices.npy", order)
    np.save(root / "tuples.npy", rows)
    np.save(root / "reaction_ids.npy", reaction)
    np.save(root / "features.npy", np.arange(len(frame)*4, dtype=np.float32).reshape(-1, 4))
    np.save(root / "product_ids.npy", np.asarray([f"{i:032x}" for i in order], dtype="S32"))
    packed = np.random.default_rng(4).integers(0, 256, size=(len(rows), 256), dtype=np.uint8)
    np.save(root / "fingerprints.npy", packed)
    np.save(root / "chemistry_filter.npy", np.ones(len(rows), bool))
    truth = np.column_stack([np.arange(len(rows))+.125, -np.arange(len(rows))-5]).astype(np.float32)
    np.save(root / "oracle_kif11.npy", truth)
    # Deliberately present a forbidden target-dependent eligibility file.
    np.save(root / "target_valid.npy", np.arange(len(rows)) % 2 == 0)
    return manifest, rows, reaction, packed, truth


class DockingRunTests(unittest.TestCase):
    def test_identity_alias_repair_is_label_free_and_warm_start_canonical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, rows, _, _, _ = make_cache(root)
            initial = run.warm_start(root, manifest, 10, 0)
            later, first = max(int(initial[0]), int(initial[1])), min(int(initial[0]), int(initial[1]))
            pids = np.load(root / "product_ids.npy")
            pids[later] = pids[first]
            np.save(root / "product_ids.npy", pids)
            manifest.update(products=len(rows), score_table=str(root / "scores.parquet"))
            (root / "manifest.json").write_text(json.dumps(manifest))
            # Exercise sorted-index duplicate discovery at chunk boundaries.
            index = root / "scores.parquet.sidx"; index.mkdir()
            ids = sorted(x.decode() for x in pids)
            np.save(index / "keys_hi.npy", np.array([int(x[:16], 16) for x in ids], np.uint64))
            np.save(index / "keys_lo.npy", np.array([int(x[16:], 16) for x in ids], np.uint64))
            original_load = np.load
            def no_labels(path, *args, **kwargs):
                self.assertNotIn("oracle_", str(path))
                self.assertNotIn("vals__", str(path))
                return original_load(path, *args, **kwargs)
            with patch.object(prepare.np, "load", side_effect=no_labels):
                result = prepare.audit_product_identity(root, chunk_size=1)
            self.assertEqual(result["aliases"], {str(later): first})
            self.assertEqual(result["unique_product_ids"], len(rows)-1)
            sampled = run.warm_start(root, manifest, 10, 0)
            self.assertEqual(sampled[0], first)
            self.assertNotIn(later, sampled)
            self.assertEqual(len(np.unique(pids[sampled])), 10)
            with self.assertRaises(ValueError):
                run.warm_start(root, manifest, len(rows), 0)
            oracle = run.BudgetOracle(root, "kif11", 10, "stored", "glide")
            with self.assertRaisesRegex(ValueError, "Noncanonical"):
                oracle.query([later])
            self.assertEqual(oracle.observed, [])
            oracle.query([first])
            self.assertEqual(oracle.observed, [first])

    def test_large_unaudited_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps({"products": 100000000}))
            with self.assertRaisesRegex(ValueError, "audit"):
                run.product_aliases(root)

    def test_budget_guards_are_atomic_and_invalid_indices_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, _, _, _, truth = make_cache(root)
            oracle = run.BudgetOracle(root, "kif11", 3, "stored", "glide")
            np.testing.assert_array_equal(oracle.query([1]), truth[[1], 0])
            for indices, error in (([1], ValueError), ([2, 2], ValueError),
                                   ([2, 3, 4], ValueError), ([-1], IndexError),
                                   ([len(truth)], IndexError)):
                with self.assertRaises(error):
                    oracle.query(indices)
                self.assertEqual(oracle.observed, [1])
                self.assertEqual(oracle.seen, {1})
            oracle.query([2, 3])
            self.assertEqual(len(oracle.observed), 3)

    def test_oracle_direction_and_failures_still_cost_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, _, _, _, truth = make_cache(root)
            truth[2] = np.nan
            np.save(root / "oracle_kif11.npy", truth)
            raw = run.BudgetOracle(root, "kif11", 3, "raw", "glide")
            np.testing.assert_array_equal(raw.query([0, 2]), [-truth[0, 1], 0])
            self.assertEqual(raw.observed, [0, 2])
            surrogate = run.BudgetOracle(root, "kif11", 2, "stored", "surrogate")
            np.testing.assert_array_equal(surrogate.query([1]), -truth[[1], 0])

    def test_warm_start_matches_historical_sampler_on_complete_grids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, rows, _, _, _ = make_cache(root)
            frame = pd.read_csv(root / "synthons.csv")
            # Independent tuple-level implementation of GlobalGAProposer's
            # initial random batch: sorted IDs, uniform reaction, slot sampling.
            lookup = {tuple(row): i for i, row in enumerate(rows)}
            rng = np.random.default_rng(3 + 104729)
            wanted = []
            while len(wanted) < 20:
                rid = manifest["reactions"][int(rng.integers(2))]
                selected = []
                for axis in manifest["layout"][rid]["slots"]:
                    axis = sorted(axis, key=lambda i: frame.synthon_id.iloc[i])
                    selected.append(int(rng.choice(axis)))
                selected += [-1] * (3-len(selected))
                index = lookup[tuple(selected)]
                if index not in wanted:
                    wanted.append(index)
            np.testing.assert_array_equal(run.warm_start(root, manifest, 20, 3), wanted)
            with self.assertRaises(ValueError):
                run.warm_start(root, manifest, len(rows)+1, 3)

    def test_runner_only_fits_charged_labels_and_never_reads_target_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, rows, reactions, packed, truth = make_cache(root)
            state = dict(oracle=None, factor_fits=0, gbm_fits=0, reads=[])
            original_oracle, original_load = run.BudgetOracle, np.load

            class ChargedArray:
                def __len__(self):
                    return len(truth)

                def __getitem__(self, key):
                    caller = inspect.currentframe().f_back
                    if caller.f_code.co_name != "query" or caller.f_code.co_filename != run.__file__:
                        raise AssertionError("Truth accessed outside charged oracle query")
                    np.testing.assert_array_equal(key[0], caller.f_locals["indices"])
                    state["reads"].append(list(key[0]))
                    return truth[key]

            def guarded_load(path, *args, **kwargs):
                name = Path(path).name
                if "target_valid" in name:
                    raise AssertionError("Docking success flag leaked into optimizer")
                if name.startswith("oracle_"):
                    caller = inspect.currentframe().f_back
                    self.assertEqual(caller.f_code.co_name, "__init__")
                    self.assertEqual(caller.f_code.co_filename, run.__file__)
                    return ChargedArray()
                return original_load(path, *args, **kwargs)

            class CapturedOracle(original_oracle):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    state["oracle"] = self

            class FactorSpy:
                def __init__(self, *args, **kwargs):
                    pass

                def fit(self, x, rid, y, **kwargs):
                    acquired = state["oracle"].observed
                    np.testing.assert_array_equal(x, rows[acquired])
                    np.testing.assert_array_equal(rid, reactions[acquired])
                    np.testing.assert_array_equal(y, state["oracle"].values)
                    state["factor_fits"] += 1
                    return {"fit_seconds": 0.}

                def predict(self, x, rid):
                    return x[:, 0].astype(float) + .3*x[:, 1] + .1*x[:, 2]

            class GBMSpy:
                def __init__(self, **kwargs):
                    pass

                def fit(self, x, y):
                    acquired = state["oracle"].observed
                    np.testing.assert_array_equal(x, np.unpackbits(packed[acquired], axis=1, bitorder="big"))
                    np.testing.assert_array_equal(y, state["oracle"].values)
                    state["gbm_fits"] += 1

                def predict(self, x):
                    return x.mean(1)

            out = root / "run"
            argv = ["run", "--cache", str(root), "--out", str(out), "--budget", "30",
                    "--batch", "10", "--device", "cpu", "--epochs", "1",
                    "--rerank", "cp64-gbm", "--diversity", ".2", "--explore", ".25"]
            with patch.object(sys, "argv", argv), patch.object(run, "BudgetOracle", CapturedOracle), \
                    patch.object(run, "FactorRegressor", FactorSpy), patch.object(np, "load", guarded_load), \
                    patch.dict(sys.modules, {"lightgbm": SimpleNamespace(LGBMRegressor=GBMSpy)}):
                run.main()
            self.assertEqual(state["factor_fits"], 4)
            self.assertEqual(state["gbm_fits"], 2)
            self.assertEqual([len(ids) for ids in state["reads"]], [10, 10, 10])
            archive = pd.read_csv(out / "observed.csv")
            self.assertEqual(len(archive), 30)
            self.assertEqual(archive.row.nunique(), 30)
            np.testing.assert_array_equal(archive.utility, truth[archive.row, 0])

    def test_prepare_aligns_permuted_incomplete_grid_to_packed_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / "data"; fps = root / "fps"; out = root / "cache"
            (data / "spaces").mkdir(parents=True); (data / "scores").mkdir()
            fpdir = fps / "synthon_space_1M_enum_fps"; fpdir.mkdir(parents=True)
            schemas = {"z": "[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]",
                       "a": "[U]-[*:1].[U]-[*:2]-[Np].[Np]-[*:3]>>[*:1]-[*:2]-[*:3]"}
            pools = {"z": [["C[U]", "CC[U]"], ["N[U]", "O[U]", "CN[U]"]],
                     "a": [["C[U]", "CC[U]"], ["[U]C[Np]", "[U]C(C)[Np]"], ["N[Np]"]]}
            frame, products, bits, expected = [], [], [], {}
            for rid, axes in pools.items():
                synthon_axes = []
                for j, smiles_axis in enumerate(axes, 1):
                    synthon_axes.append([])
                    for smi in smiles_axis:
                        sid = 100+len(frame)
                        synthon_axes[-1].append((sid, smi))
                        frame.append(dict(reaction_id=rid, position=j, synthon_id=sid, smiles=smi))
                for candidate in itertools.product(*synthon_axes):
                    smi = assemble([x[1] for x in candidate], schemas[rid])
                    idx = len(products); pid = f"{idx:032x}"
                    expected[pid] = idx
                    record = dict(product_id=pid, reaction_id=rid, smiles=smi, passes_exp27_exact_filter=True)
                    record.update({f"synthon{j+1}_id": candidate[j][0] if j < len(candidate) else None for j in range(3)})
                    for target in ("kif11", "pyrd", "tgfr1"):
                        record[f"{target}_glide_le"] = idx+.1
                        record[f"{target}_glide_raw"] = -idx-5.
                    products.append(record)
                    b = np.zeros(2048, np.uint8)
                    DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(Chem.MolFromSmiles(smi), fpSize=2048), b)
                    bits.append(b)
            pd.DataFrame(frame).to_csv(data / "spaces/synthon_space_1M.synthons.tsv", sep="\t", index=False)
            pd.DataFrame([dict(reaction_id=r, Reaction=s) for r, s in schemas.items()]).to_csv(data / "spaces/reactions.tsv", sep="\t", index=False)
            # Drop one Cartesian product and scramble all remaining table rows.
            selected = pd.DataFrame(products).drop(index=2).sample(frac=1, random_state=23)
            selected.to_parquet(data / "scores/glide_1M.parquet", index=False)
            packed = np.packbits(bits, axis=1, bitorder="big")
            with gzip.open(fpdir / "fps_0000.npy.gz", "wb") as handle:
                np.save(handle, packed)
            with gzip.open(fpdir / "fps_0000.json.gz", "wt") as handle:
                json.dump(dict(chunk_start_index=0), handle)
            (fpdir / "metadata.json").write_text(json.dumps(dict(fp_type="rdkit", fp_size=2048,
                                                                   total_smiles=len(products), total_invalid=0, n_chunks=1)))
            argv = ["prepare", "--data", str(data), "--fingerprints", str(fps), "--out", str(out)]
            with patch.object(sys, "argv", argv):
                prepare.main()
            wanted = np.asarray([expected[pid] for pid in selected.product_id])
            np.testing.assert_array_equal(np.load(out / "enum_indices.npy"), wanted)
            np.testing.assert_array_equal(np.load(out / "fingerprints.npy"), packed[wanted])
            self.assertEqual(json.loads((out / "manifest.json").read_text())["bitorder"], "big")
            self.assertFalse((out / "target_valid.npy").exists())


if __name__ == "__main__":
    unittest.main()
