"""Sample assembled products, label them with a teacher and encode capped synthons."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chemistry import DATA, cap_synthon, sample_products, slot_groups
from teachers import RDKitTeacher, OpenADMETTeacher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--space", type=Path, default=DATA / "synthon_space_1M.tsv")
    p.add_argument("--reactions", type=Path, default=DATA / "reactions.tsv")
    p.add_argument("--teacher", choices=["rdkit", "openadmet"], default="rdkit")
    p.add_argument("--repository", default="herg-chemeleon-baseline")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--foundation-root", type=Path)
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=38026)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/rdkit"))
    args = p.parse_args()
    if args.samples < 30:
        p.error("Use at least 30 samples for train/validation/test splits")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame = pd.read_csv(
        args.space, sep="\t", dtype={"synthon_id": str, "reaction_id": str}
    )
    frame = frame[["smiles", "synthon_id", "position", "reaction_id"]].copy()
    capped, invalid = [], []
    for i, smiles in enumerate(frame.smiles):
        try:
            capped.append(cap_synthon(smiles))
        except (ValueError, RuntimeError):
            capped.append(None)
            invalid.append(i)
    frame["capped_smiles"] = capped
    frame = frame.dropna().reset_index(drop=True)
    # Reject an incomplete reaction instead of silently changing its arity.
    schema_frame = pd.read_csv(args.reactions, sep="\t", dtype={"reaction_id": str})
    schema_frame = schema_frame.set_index("reaction_id")
    groups = slot_groups(frame)
    groups = {
        r: slots
        for r, slots in groups.items()
        if r in schema_frame.index
        and len(slots) == int(schema_frame.loc[r, "components"])
    }
    frame = frame[frame.reaction_id.isin(groups)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete supported reactions remain")
    products, assembly_failures = sample_products(
        frame, schema_frame.Reaction.to_dict(), args.samples, args.seed
    )
    preparation_seconds = time.perf_counter() - started
    start = time.perf_counter()
    teacher = (
        RDKitTeacher()
        if args.teacher == "rdkit"
        else OpenADMETTeacher(
            args.repository, args.device, args.model_dir, args.foundation_root
        )
    )
    model_load_seconds = time.perf_counter() - start
    unique, inverse = np.unique(frame.capped_smiles.to_numpy(str), return_inverse=True)
    start = time.perf_counter()
    embeddings = teacher.encode(unique)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y = teacher.predict(products.smiles.tolist())
    teacher_seconds = time.perf_counter() - start
    tuples = products[["row_1", "row_2", "row_3"]].to_numpy(np.int64)
    # Canonical-product split: duplicate products cannot leak between splits.
    residues = np.asarray(
        [
            int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 10
            for s in products.smiles
        ]
    )
    split = np.where(
        residues < 2, "test", np.where(residues == 2, "validation", "train")
    )
    if any(np.sum(split == name) < 2 for name in ("train", "validation", "test")):
        raise ValueError("Too few examples in a split; increase --samples")
    products["split"] = split
    for i, name in enumerate(teacher.targets):
        products[name] = y[:, i]
    frame.to_csv(args.out / "synthons.csv", index=False)
    products.to_csv(args.out / "products.csv", index=False)
    np.savez_compressed(
        args.out / "dataset.npz",
        embeddings=embeddings.astype(np.float32),
        row_to_embedding=inverse,
        tuples=tuples,
        targets=y,
        split=split,
    )
    manifest = {
        **teacher.provenance,
        "targets": teacher.targets,
        "seed": args.seed,
        "synthon_rows": len(frame),
        "unique_capped_synthons": len(unique),
        "sampled_products": len(products),
        "uncappable_rows_excluded": len(invalid),
        "assembly_failures": assembly_failures,
        "split_counts": {
            s: int(np.sum(split == s)) for s in ("train", "validation", "test")
        },
        "space_sha256": hashlib.sha256(args.space.read_bytes()).hexdigest(),
        "timing_seconds": {
            "capping_sampling_assembly": preparation_seconds,
            "teacher_load": model_load_seconds,
            "synthon_encoding": encode_seconds,
            "product_teacher": teacher_seconds,
        },
        "device": args.device,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "torch", "rdkit")
        },
    }
    (args.out / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
