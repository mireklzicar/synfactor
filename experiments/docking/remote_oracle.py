"""Standalone Dock-12 bridge: read-only preflight unless --execute is explicit.

Proposal JSON: target, seed, run_id, batch_id, product_ids (ordered unique IDs).
Optional submitted_smiles maps IDs to independently assembled structures, which
must agree with native remote assembly. Run outputs stay in the fixed new root.
An unresolved DOCKING_STARTED marker prohibits automatic resubmission.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

ROOT = Path("/home/ubuntu/synfactor_dock12_20261116")
REPO = Path("/home/mirekl/SynthonBench_ICML/synthonbench_v2")
LIBRARY = Path("/home/mirekl/TWIST_Navigator/freedom5_skach")
HARNESS = Path("/home/mirekl/TWIST_Navigator/dmc-harness-original/src")
SUITE = Path("/opt/schrodinger2026-1")
CONTAINRE = Path("/home/ubuntu/dmc/ContainRE")
SETTINGS = dict(precision="HTVS", forcefield="OPLS_2005", n_poses=1,
                ligprep_cpu=1, glide_cpu=1, max_stereoisomers=1, ph=7.,
                epik=True, epik_mode="classic")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def preflight(target, seed):
    sys.path[:0] = [str(REPO / "src"), str(HARNESS)]
    from synthon_assembler import assembly as native_assembly, index as native_index
    from dmc_navigator_prod.core import space as production_space
    from dmc_navigator_prod.api import models as production_models
    from dmc_navigator_prod.chemistry import synthon_assembler_backend as production_assembly
    manifest_path = REPO / "configs/targets_dock12.json"
    targets = {r["target_id"]: r for r in json.loads(manifest_path.read_text())["targets"]}
    if target not in targets or seed not in (0, 1, 2):
        raise ValueError("Unknown Dock-12 target or seed")
    grids = list((REPO / "targets" / target / "glide").glob("*.zip"))
    if len(grids) != 1:
        raise ValueError("Expected exactly one existing target grid")
    grid = grids[0]
    ids_path = REPO / f"artifacts/seeds/freedom_296b/seed_{seed}/selected_10k/subselections/budget_01000.ids"
    candidates_path = ids_path.parents[1] / "candidates.csv"
    matrix = REPO / f"artifacts/dock12_seed_screen/freedom/seed_{seed}/results_long.csv"
    ids = ids_path.read_text().splitlines()
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Warm cohort must contain exactly 1000 distinct IDs")
    wanted = set(ids)
    with matrix.open() as handle:
        rows = {r["product_id"]: r for r in csv.DictReader(handle)
                if r["target_id"] == target and r["product_id"] in wanted}
    if set(rows) != wanted:
        raise ValueError("Missing target-specific warm observations")
    baseline_path = REPO / f"artifacts/navigator_glide/nav_freedom_seed{seed}_{target}_ga_dcso_v14_screening/run_manifest.json"
    baseline = json.loads(baseline_path.read_text())
    grid_hash = digest(grid)
    if baseline["grid"]["sha256"] != grid_hash:
        raise ValueError("Current grid differs from matched Navigator archive")
    hashes = {str(p): digest(p) for p in [manifest_path, ids_path, candidates_path, matrix, baseline_path,
                                        LIBRARY / "reactions.txt", LIBRARY / "syntons.txt",
                                        HARNESS / "glide_harness/coupled.py",
                                        REPO / "scripts/run_navigator_glide_cell.py", Path(__file__).resolve(),
                                        Path(native_assembly.__file__), Path(native_index.__file__),
                                        Path(production_space.__file__), Path(production_models.__file__),
                                        Path(production_assembly.__file__), LIBRARY / "synthon_index.pkl"]}
    return {"target": target, "seed": seed, "glide_site_id": targets[target]["glide_site_id"],
            "grid": str(grid), "grid_sha256": grid_hash, "settings": SETTINGS,
            "warm_count": len(ids), "warm_ordered_ids_sha256": digest(ids_path),
            "warm_candidates_file": str(candidates_path),
            "warm_valid_count": sum(str(r["valid"]).lower() in ("1", "true", "yes") for r in rows.values()),
            "matched_baseline": str(baseline_path), "source_sha256": hashes,
            "free_disk_gib": shutil.disk_usage(ROOT.parent).free / 1024**3}, wanted


def execute(args, proposal, protocol, warm_ids):
    for field in ("run_id", "batch_id"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(proposal[field])):
            raise ValueError(f"Invalid {field}")
    ids = proposal["product_ids"]
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids):
        raise ValueError("Need distinct ordered product ID strings")
    if args.smoke:
        if not proposal["run_id"].startswith("smoke_") or len(ids) > 16:
            raise ValueError("Smoke must be a separate smoke_ run, at most 16 products")
    elif len(ids) != 1000 or set(ids) & warm_ids:
        raise ValueError("Adaptive batches require 1000 new IDs, excluding charged warm cohort")
    cell = ROOT / proposal["run_id"]
    batch = cell / proposal["batch_id"]
    batch.mkdir(parents=True, exist_ok=True)
    with (cell / "oracle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {k:v for k,v in protocol.items() if k != "free_disk_gib"}
        cell_protocol = cell / "protocol.json"
        if cell_protocol.exists() and json.loads(cell_protocol.read_text()) != identity:
            raise ValueError("Cell target/seed/space/warm/oracle protocol changed")
        if not cell_protocol.exists():
            write(cell_protocol, identity)
        request_hash = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()
        if (batch / "request.json").exists():
            previous = json.loads((batch / "request.json").read_text())
            if previous["request_sha256"] != request_hash or previous["protocol"] != protocol:
                # Free disk is telemetry, not part of oracle identity.
                old = {k:v for k,v in previous["protocol"].items() if k != "free_disk_gib"}
                new = {k:v for k,v in protocol.items() if k != "free_disk_gib"}
                if previous["request_sha256"] != request_hash or old != new:
                    raise ValueError("Batch request/protocol changed")
        if (batch / "normalized.json").exists():
            print((batch / "normalized.json").read_text())
            return
        if any(p.exists() and not (p.parent / "normalized.json").exists()
               for p in cell.glob("*/DOCKING_STARTED.json")):
            raise RuntimeError("Unresolved docking exists; reconcile retained work, never resubmit automatically")
        previous_ids = set()
        for p in cell.glob("*/normalized.json"):
            previous_ids.update(r["product_id"] for r in json.loads(p.read_text())["rows"])
        if previous_ids.intersection(ids) or (not args.smoke and len(previous_ids)+len(ids)>9000):
            raise ValueError("Duplicate submissions or adaptive budget exceeded")
        if protocol["free_disk_gib"] < 12:
            raise RuntimeError("Disk guard: fewer than 12 GiB free")
        sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(HARNESS)]
        from dmc_navigator_prod.api.models import SpaceSpec
        from dmc_navigator_prod.core.space import load_space
        from glide_harness.coupled import SandboxSpec, dock_coupled
        from run_navigator_glide_cell import parse_scores, archive_and_prune, drop_reuse_container
        from rdkit import Chem
        space = load_space(SpaceSpec(reactions_path=str(LIBRARY / "reactions.txt"),
                                    synthons_path=str(LIBRARY / "syntons.txt"),
                                    assembly_backend="synthon_assembler",
                                    synthon_assembler_cache_path=str(LIBRARY / "synthon_index.pkl"),
                                    strip_connectors=False, sampling_regime="product_weighted"))
        routes = []
        for pid in ids:
            rid, *synthon_ids = pid.split("____")
            route = space.make_candidate(rid, tuple(synthon_ids))
            if route.product_id != pid:
                raise ValueError("Native route identity differs from submitted ID")
            routes.append(route)
        products = space.assemble_many(routes)
        submitted = []
        for pid, product in zip(ids, products):
            smiles = getattr(product, "smiles", None) if getattr(product, "status", None) == "valid" else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            submitted.append((pid, None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)))
        supplied = proposal.get("submitted_smiles", {})
        warm_comparisons = 0
        if args.smoke:
            with Path(protocol["warm_candidates_file"]).open() as handle:
                warm_structures = {r["product_id"]: r["smiles"] for r in csv.DictReader(handle)
                                   if r["product_id"] in ids and r["product_id"] in warm_ids}
            supplied = {**warm_structures, **supplied}
            warm_comparisons = len(warm_structures)
        for pid, smiles in submitted:
            if pid in supplied:
                mol = Chem.MolFromSmiles(supplied[pid]) if supplied[pid] else None
                if (None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)) != smiles:
                    raise ValueError(f"Local/remote submitted structure differs for {pid}")
        write(batch / "request.json", {"request_sha256": request_hash, "proposal": proposal,
                                      "protocol": protocol, "assembly": {
                                          "backend": "unchanged Navigator0.4.0 synthon_assembler",
                                          "index": str(LIBRARY / "synthon_index.pkl"),
                                          "index_sha256": protocol["source_sha256"][str(LIBRARY / "synthon_index.pkl")]}})
        write(batch / "submitted.json", submitted)
        ligands = [(pid, smiles) for pid, smiles in submitted if smiles]
        if not ligands:
            raise RuntimeError("No assembled products; retained submissions require diagnosis")
        work = batch / "work"
        reuse = "synfactor-" + hashlib.sha256(str(cell).encode()).hexdigest()[:20]
        sandbox = SandboxSpec(suite=SUITE, containre_repo=CONTAINRE, work_dir=work,
                              cpus=32, mem_mb=38000, pids=16384, disk_mb=50000,
                              wall_time_limit_s=900, reuse_container=True, reuse_key=reuse)
        write(batch / "DOCKING_STARTED.json", {"unix_time": time.time(), "submitted": len(ligands),
                                              "request_sha256": request_hash})
        started = time.perf_counter()
        try:
            result = dock_coupled(ligands, protocol["glide_site_id"], grid_zip=Path(protocol["grid"]),
                                  workers=32, chunk_size=32, sandbox=sandbox, run=True,
                                  verbose=False, **SETTINGS)
            rows = parse_scores(work, ids)
            submitted_map = dict(submitted)
            for row in rows:
                row["smiles"] = submitted_map[row["product_id"]]
                score = row["score"]
                if score is None or not math.isfinite(score) or abs(score) >= 100:
                    row.update(valid=False, score=None, failure_reason="nonfinite_missing_or_sentinel_score")
            if not result.ok or not any(r["valid"] for r in rows):
                raise RuntimeError("Harness failed or no finite valid scores; retained work must be reconciled")
            poses = [p for p in (work / "out").rglob("*") if p.is_file() and
                     p.name.endswith((".maegz", ".mae", ".sdf", ".sdf.gz", ".pdb"))]
            with tarfile.open(batch / "poses.tar.gz", "w:gz") as archive:
                for path in poses:
                    archive.add(path, arcname=str(path.relative_to(work / "out")), recursive=False)
            # Preserve selected original representations and every raw score;
            # prune scratch only after independent pose retention succeeds.
            archive_and_prune(work, batch / "raw_scores.csv.gz")
            payload = {"target": proposal["target"], "seed": proposal["seed"],
                       "run_id": proposal["run_id"], "batch_id": proposal["batch_id"],
                       "request_sha256": request_hash, "rows": rows,
                       "logical_submissions": len(ids), "new_dockings": len(ligands),
                       "seconds": time.perf_counter()-started, "pose_files": len(poses),
                       "warm_structure_comparisons": warm_comparisons,
                       "smoke_only": args.smoke, "protocol": protocol}
            write(batch / "normalized.json", payload)
            print(json.dumps(payload, allow_nan=False))
        finally:
            drop_reuse_container(reuse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="kif11_spacehasten")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text()) if args.proposal else None
    target, seed = (proposal["target"], int(proposal["seed"])) if proposal else (args.target, args.seed)
    protocol, warm_ids = preflight(target, seed)
    if not args.execute:
        print(json.dumps({"preflight_only": True, **protocol}, indent=2))
    elif proposal is None:
        parser.error("--execute requires --proposal")
    else:
        execute(args, proposal, protocol, warm_ids)


if __name__ == "__main__":
    main()
