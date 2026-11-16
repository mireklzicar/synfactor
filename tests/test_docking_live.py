"""Live acquisition budget/identity and transport-checkpoint contract tests."""
import hashlib
import copy
import importlib.util
import json
import shlex
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if importlib.util.find_spec("pyarrow") is None:
    raise unittest.SkipTest("Optional full-space integration tests require pyarrow")

from experiments.docking import live
from experiments.docking.fullspace import FullSpace


class FakeSpace:
    features = np.ones((25000, 2), np.float32)
    reactions = ["r"]
    grids = [dict(reaction_index=0, slots=[np.arange(25000), np.array([0])])]
    manifest = dict(source_sha256={"reactions.txt": "reaction-sha", "syntons.txt": "synthon-sha"},
                    products=25000)

    def __init__(self):
        self.cursor = 9000

    def decode(self, ids):
        return np.asarray([[int(pid[1:]), 0, -1] for pid in ids]), np.zeros(len(ids), np.int64)

    def ids(self, rows, rids):
        return [f"p{row[0]}" for row in rows]

    def sample(self, count, rng, excluded=()):
        excluded = set(excluded)
        rows = []
        while len(rows) < count:
            row = [self.cursor, 0, -1]; self.cursor += 1
            if (0, *row) not in excluded:
                rows.append(row)
        return np.asarray(rows), np.zeros(count, np.int64)


def history():
    return [dict(product_id=f"p{i}", score=-float(i+1) if i % 7 else None,
                 status="valid" if i % 7 else "failed", smiles="C", charged=True,
                 iteration=-1, source="charged_shared_warm") for i in range(1000)]


class LiveTests(unittest.TestCase):
    def test_remote_path_retains_uv_and_system_library_utilities(self):
        oracle = live.RemoteOracle("host", "/key", "/python", "/root")
        with patch.object(live.subprocess,"run") as run:
            oracle.command(["/python","--version"])
        invocation = shlex.split(run.call_args.args[0][-1])
        paths = invocation[1].removeprefix("PATH=").split(":")
        self.assertTrue({"/home/ubuntu/.local/bin","/usr/bin","/usr/sbin","/sbin"}.issubset(paths))

    def test_checkpoint_rejects_changed_warm_duplicates_and_inconsistent_rounds(self):
        warm = history()
        adaptive = [dict(product_id=f"p{i}", score=-1., status="valid", smiles="C",
                         iteration=1, source="live_glide", charged=True) for i in range(1000, 2000)]
        state = dict(history=warm+adaptive, rounds=[dict(iteration=1, calls=2000)])
        live.validate_state(state, warm)
        mutations = [lambda s: s["history"].pop(),
                     lambda s: s["history"][0].update(score=-123.),
                     lambda s: s["history"][-1].update(product_id="p1000"),
                     lambda s: s["rounds"][0].update(calls=1999),
                     lambda s: s["rounds"][0].update(iteration=2),
                     lambda s: s["history"][-1].update(iteration=0)]
        for mutation in mutations:
            invalid = copy.deepcopy(state)
            mutation(invalid)
            with self.assertRaises(ValueError):
                live.validate_state(invalid, warm)

    def test_protocol_identity_ignores_only_telemetry(self):
        protocol = dict(target="tyk2", grid_sha256="grid", settings=dict(precision="HTVS"),
                        source_sha256={"oracle.py": "source"}, free_disk_gib=90., preflight_only=True)
        expected = live.protocol_digest(protocol)
        self.assertEqual(live.protocol_digest({**protocol, "free_disk_gib": 20., "preflight_only": False}), expected)
        for changed in ({"grid_sha256": "other"}, {"target": "pyrd"},
                        {"settings": dict(precision="SP")}, {"source_sha256": {"oracle.py": "changed"}}):
            self.assertNotEqual(live.protocol_digest({**protocol, **changed}), expected)

    def test_acquisition_fits_only_charged_labels_and_returns_700_200_100(self):
        space, observed = FakeSpace(), history()
        rows, rids = space.decode([r["product_id"] for r in observed])
        labels = np.asarray([-r["score"] if r["status"] == "valid" else 0 for r in observed], np.float32)
        state = dict(fits=0, information=0)
        test = self

        class Model:
            def __init__(self, features, n_reactions, **kwargs):
                test.assertIs(features, space.features)
                test.assertEqual(kwargs["seed"], 17)

            def fit(self, x, reaction, y, **kwargs):
                np.testing.assert_array_equal(x, rows)
                np.testing.assert_array_equal(reaction, rids)
                np.testing.assert_array_equal(y, labels)
                state["fits"] += 1
                return {"epochs": 1}

            def predict(self, x, reaction):
                return x[:, 0].astype(np.float32) / 1000

            def acquisition_features(self, x, reaction):
                return np.column_stack([reaction, x]).astype(np.float32)

        def sweep(model, grids, **kwargs):
            np.testing.assert_array_equal(kwargs["excluded_tuples"], rows)
            np.testing.assert_array_equal(kwargs["excluded_reaction_ids"], rids)
            candidates = np.column_stack([np.arange(1000, 21000), np.zeros(20000, int), -np.ones(20000, int)])
            return dict(tuples=candidates, reaction_ids=np.zeros(20000, int),
                        scores=np.arange(20000, dtype=np.float32), diagnostics={"complete": True})

        def diversity(ids, scores, count, packed, bitorder, references, strength, device, **kwargs):
            test.assertEqual(count, 700)
            test.assertIsNone(packed)
            reference_rows = kwargs["synthon_rows"][references]
            test.assertTrue(set(reference_rows[:, 0]).issubset(set(rows[:, 0])))
            return np.arange(count)

        class Information:
            def __init__(self, features, y, predicted, **kwargs):
                np.testing.assert_array_equal(features[:, 1:], rows)
                np.testing.assert_array_equal(y, labels)
                np.testing.assert_allclose(predicted, rows[:, 0]/1000, atol=1e-7)

            def select(self, features, predictions, count, **kwargs):
                test.assertEqual(count, 200)
                condition = kwargs["condition_features"]
                test.assertEqual(len(condition), 700)
                test.assertEqual(len(np.unique(features, axis=0)), len(features))
                test.assertFalse(set(features[:, 1]) & set(rows[:, 0]))
                test.assertFalse(set(features[:, 1]) & set(condition[:, 1]))
                state["information"] += 1
                return np.arange(count), {"selected_count": count}

        with patch.object(live, "FactorRegressor", Model), patch.object(live, "stream_topk", sweep), \
                patch.object(live, "diverse_select", diversity), patch.object(live, "LinearInformation", Information), \
                patch.object(live.torch.cuda, "empty_cache"):
            ids, diagnostics = live.propose(space, observed, dict(seed=0, information_fraction=.2,
                                                                 sweep_seconds=300), 1)
        self.assertEqual(len(ids), 1000)
        self.assertEqual(len(set(ids)), 1000)
        self.assertFalse(set(ids) & {r["product_id"] for r in observed})
        self.assertEqual(state, dict(fits=1, information=1))
        self.assertEqual([diagnostics[k] for k in ("exploit_count", "information_count", "random_count")], [700, 200, 100])

    def test_fullspace_mixed_arity_roundtrip_and_unique_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = pd.DataFrame([
                ("a", 1, "x"), ("a", 1, "y"), ("a", 2, "q"), ("a", 2, "z"),
                ("b", 1, "x"), ("b", 2, "u"), ("b", 2, "v"), ("b", 3, "s"), ("b", 3, "t")],
                columns=["reaction_id", "position", "synthon_id"])
            frame.to_parquet(root/"synthons.parquet", index=False)
            np.save(root/"features.npy", np.zeros((len(frame), 516), np.float32))
            (root/"manifest.json").write_text(json.dumps(dict(reactions=["a", "b"], grids=[
                dict(reaction_index=0, slots=[[0, 2], [2, 4]], products=4),
                dict(reaction_index=1, slots=[[4, 5], [5, 7], [7, 9]], products=4)])))
            space = FullSpace(root)
            ids = ["a____x____q", "b____x____v____t"]
            rows, rids = space.decode(ids)
            self.assertEqual(space.ids(rows, rids), ids)
            self.assertEqual(rows.tolist(), [[0, 2, -1], [4, 6, 8]])
            excluded = {(int(rid), *row) for row, rid in zip(rows, rids)}
            sampled, sampled_rids = space.sample(6, np.random.default_rng(7), excluded)
            sampled_ids = space.ids(sampled, sampled_rids)
            self.assertEqual(len(set(sampled_ids)), 6)
            self.assertFalse(set(sampled_ids) & set(ids))
            np.testing.assert_array_equal(space.decode(sampled_ids)[0], sampled)
            for invalid in ("a____x____t", "b____x____v", "unknown____x____q"):
                with self.assertRaises(ValueError):
                    space.decode([invalid])

    def test_cached_remote_response_validates_hash_order_and_smoke_without_ssh(self):
        proposal = dict(run_id="cell", batch_id="batch_01", target="tyk2", seed=0,
                        product_ids=["a", "b"])
        expected = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
        payload = dict(request_sha256=expected, rows=[dict(product_id="a"), dict(product_id="b")], smoke_only=False)
        oracle = live.RemoteOracle("unused-host", "unused-key", "python", "/unused-root")
        with tempfile.TemporaryDirectory() as temp, patch.object(live.subprocess, "run") as process:
            folder = Path(temp)
            for update in ({}, {"request_sha256": "wrong"}, {"smoke_only": True},
                           {"rows": [dict(product_id="b"), dict(product_id="a")]}):
                (folder/"normalized.json").write_text(json.dumps({**payload, **update}))
                if update:
                    with self.assertRaises(ValueError):
                        oracle.query(proposal, folder)
                else:
                    self.assertEqual(oracle.query(proposal, folder), payload)
            process.assert_not_called()
            with self.assertRaises(ValueError):
                oracle.query({**proposal, "product_ids": ["a", "c"]}, folder)

    def test_changed_configuration_blocks_resume_before_oracle_query(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(out=root, seed=0, information_fraction=.2, benchmark_repo=root/"bench", sweep_seconds=300,
                                   max_cell_seconds=3600)
            cell = root/"sf_freedom_seed0_tyk2_info20"; cell.mkdir()
            (cell/"config.json").write_text(json.dumps({"stale": True}))
            warm_hashes = dict(ids_sha256="ids", matrix_sha256="matrix", csv_sha256="csv", candidates_sha256="candidates")
            matrix = args.benchmark_repo/"artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv"
            protocol = dict(warm_ordered_ids_sha256="ids", grid_sha256="grid", source_sha256={
                "/home/mirekl/TWIST_Navigator/freedom5_skach/reactions.txt": "reaction-sha",
                "/home/mirekl/TWIST_Navigator/freedom5_skach/syntons.txt": "synthon-sha", str(matrix): "matrix"})
            oracle = SimpleNamespace(preflight=lambda target, seed: protocol,
                                     query=lambda *a: self.fail("Changed resume reached the remote oracle"))
            with patch.object(live, "warm_start", return_value=(history(), warm_hashes)):
                with self.assertRaisesRegex(ValueError, "configuration/source changed"):
                    live.run_cell(args, FakeSpace(), oracle, "tyk2")
            self.assertFalse((cell/"state.json").exists())

    def test_fresh_cell_finishes_all_batches_without_mutating_warm_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(out=root, seed=0, information_fraction=.2, benchmark_repo=root/"bench",
                                   sweep_seconds=300, max_cell_seconds=3600)
            warm = history(); original = copy.deepcopy(warm)
            hashes = dict(ids_sha256="ids", matrix_sha256="matrix", csv_sha256="csv", candidates_sha256="candidates")
            matrix = args.benchmark_repo/"artifacts/dock12_seed_screen/freedom/seed_0/results_long.csv"
            protocol = dict(warm_ordered_ids_sha256="ids", grid_sha256="grid", source_sha256={
                "/home/mirekl/TWIST_Navigator/freedom5_skach/reactions.txt": "reaction-sha",
                "/home/mirekl/TWIST_Navigator/freedom5_skach/syntons.txt": "synthon-sha", str(matrix): "matrix"})
            calls = []
            def query(proposal, folder):
                calls.append(proposal["batch_id"])
                return dict(protocol=protocol, seconds=1., rows=[
                    dict(product_id=pid, smiles="C", score=-1., valid=True) for pid in proposal["product_ids"]])
            def propose(space, acquired, config, iteration):
                return [f"p{i}" for i in range(iteration*1000, (iteration+1)*1000)], dict(search_seconds=1.)
            oracle = SimpleNamespace(preflight=lambda *a: protocol, query=query)
            with patch.object(live, "warm_start", return_value=(warm, hashes)), patch.object(live, "propose", side_effect=propose), patch("builtins.print"):
                live.run_cell(args, FakeSpace(), oracle, "tyk2")
                live.run_cell(args, FakeSpace(), oracle, "tyk2")
            self.assertEqual(warm, original)
            self.assertEqual(calls, [f"batch_{i:02d}" for i in range(1, 10)])
            cell = root/"sf_freedom_seed0_tyk2_info20"
            self.assertTrue((cell/"CELL_DONE").exists())
            saved = json.loads((cell/"state.json").read_text())
            live.validate_state(saved, warm)
            self.assertEqual(len(saved["history"]), 10000)


if __name__ == "__main__":
    unittest.main()
