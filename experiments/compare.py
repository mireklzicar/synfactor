"""Dense small-grid comparison against the same full-product teachers.

Run from the repo root: python -m experiments.compare --device cuda
All basis-product structures are reserved for training, including duplicates
across routes. Test scores measure unseen combinations of known synthons.
"""

import argparse
import copy
import gc
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski
from scipy.stats import spearmanr

from chemistry import DATA, assemble, cap_synthon, slot_groups
from model import SynFactor
from teachers import OpenADMETTeacher, RDKitTeacher, REVISIONS
from .baselines import basis_indices, shi_predict

DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "HAC": lambda m: m.GetNumHeavyAtoms(),
    "HBA": Lipinski.NumHAcceptors, "HBD": Lipinski.NumHDonors,
    "TPSA": Descriptors.TPSA, "CrippenLogP": Crippen.MolLogP,
}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare(out, seed):
    if (out / "products.csv").exists():
        design = json.loads((out / "design.json").read_text())
        if design["seed"] != seed:
            raise ValueError("Cached split uses a different seed; use a new output directory")
        return
    start = time.perf_counter()
    frame = pd.read_csv(DATA / "synthon_space_1M.tsv", sep="\t", dtype={"reaction_id": str, "synthon_id": str})
    schemas = pd.read_csv(DATA / "reactions.tsv", sep="\t", dtype={"reaction_id": str}).set_index("reaction_id").Reaction.to_dict()
    groups, rng = slot_groups(frame), np.random.default_rng(seed)
    dis = sorted(rng.choice([r for r, ss in groups.items() if len(ss) == 2], 6, replace=False))
    selected = dis + sorted(r for r, ss in groups.items() if len(ss) == 3)
    synthons, records, failed = [], [], []
    for rid in selected:
        slots = []
        for j, candidates in enumerate(groups[rid]):
            valid = []
            for row in rng.permutation(candidates):
                entry = frame.iloc[row].to_dict()
                try:
                    entry["capped_smiles"] = cap_synthon(entry["smiles"])
                except (ValueError, RuntimeError):
                    continue
                valid.append(entry)
                if len(valid) == (20 if len(groups[rid]) == 2 else 10):
                    break
            # The reference is the smallest selected synthon, fixed before labels.
            valid.sort(key=lambda e: (Chem.MolFromSmiles(e["capped_smiles"]).GetNumHeavyAtoms(), e["smiles"]))
            ids = []
            for local, entry in enumerate(valid):
                entry["local_index"] = local
                ids.append(len(synthons))
                synthons.append(entry)
            slots.append(ids)
        for local in itertools.product(*(range(len(x)) for x in slots)):
            ids = [slots[j][i] for j, i in enumerate(local)]
            try:
                smiles = assemble([synthons[i]["smiles"] for i in ids], schemas[rid])
            except (ValueError, RuntimeError):
                failed.append((rid, local))
                continue
            records.append(dict(reaction_id=rid, smiles=smiles,
                                **{f"row_{j+1}": ids[j] if j < len(ids) else -1 for j in range(3)},
                                **{f"local_{j+1}": local[j] if j < len(local) else -1 for j in range(3)}))
    if failed:
        raise ValueError(f"Dense-grid audit requires all products; failed {failed[:5]}")
    products = pd.DataFrame(records)
    basis, refs = basis_indices(products[[f"local_{j}" for j in (1, 2, 3)]].values, products.reaction_id)
    basis_rows = np.unique(np.concatenate([basis[basis >= 0], refs]))
    basis_smiles = set(products.iloc[basis_rows].smiles)
    residues = [int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()[:8], 16) % 10 for s in products.smiles]
    products["split"] = ["train" if s in basis_smiles or r >= 3 else "test" if r < 2 else "validation" for s, r in zip(products.smiles, residues)]
    products["basis"] = products.smiles.isin(basis_smiles)
    assert products.groupby("smiles").split.nunique().max() == 1
    assert not products.loc[products.split != "train", "basis"].any()
    pd.DataFrame(synthons).to_csv(out / "synthons.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    np.savez_compressed(out / "basis.npz", basis=basis, refs=refs)
    dump(out / "design.json", {
        "seed": seed, "space_sha256": hashlib.sha256((DATA / "synthon_space_1M.tsv").read_bytes()).hexdigest(),
        "reactions": selected, "tuples": len(products), "unique_products": products.smiles.nunique(),
        "synthon_rows": len(synthons), "basis_unique_products": len(basis_smiles),
        "split_unique_products": products.drop_duplicates("smiles").split.value_counts().to_dict(),
        "preparation_seconds": time.perf_counter() - start,
    })


def fit_cp(x, y, rows, split, seed, device, epochs=250):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    rows = torch.tensor(rows, device=device)
    present = rows >= 0
    train, val = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    model = SynFactor(x.shape[1], y.shape[1]).to(device)
    used = torch.unique(rows[train][present[train]])
    model.x_mean.copy_(x[used].mean(0))
    model.x_scale.copy_(x[used].std(0).clamp_min(1e-4))
    model.y_mean.copy_(y[train].mean(0))
    model.y_scale.copy_(y[train].std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    best, stale, history = float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(train), max(1, len(train) // 256)):
            pred = model(x[rows[ids].clamp_min(0)], present[ids])
            loss = torch.nn.functional.huber_loss((pred - model.y_mean) / model.y_scale, (y[ids] - model.y_mean) / model.y_scale)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x[rows[val].clamp_min(0)], present[val])
            value = (((pred - y[val]) / model.y_scale) ** 2).mean().item()
        history.append(value)
        if value < best:
            best, stale, state = value, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 35:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[rr.clamp_min(0)], rr >= 0) for rr in rows.split(256)]).cpu().numpy()
    return pred, {"epochs": epoch + 1, "best_validation_nmse": best, "validation_history": history}, state


def additive_fit(y, rows, split):
    """Train-only ridge slot lookup, lambda selected by validation, no embeddings."""
    n = rows.max() + 1
    design = np.zeros((len(rows), n + 1))
    design[:, -1] = 1
    for j in range(3):
        keep = rows[:, j] >= 0
        design[np.flatnonzero(keep), rows[keep, j]] = 1
    train, val = split == "train", split == "validation"
    scale = np.maximum(y[train].std(0), 1e-6)
    mean = y[train].mean(0)
    gram = design[train].T @ design[train]
    rhs = design[train].T @ ((y[train] - mean) / scale)
    best, chosen = float("inf"), None
    for penalty in (1e-6, .001, .1, 1., 10.):
        weights = np.linalg.solve(gram + penalty * np.eye(n + 1), rhs)
        pred = design @ weights * scale + mean
        loss = np.mean(((pred[val] - y[val]) / scale) ** 2)
        if loss < best:
            best, chosen, result = loss, penalty, pred
    return result, {"ridge_penalty": chosen, "validation_nmse": best}


def metric_rows(y, pred, names, method, products):
    result = []
    unique = ~products.smiles.duplicated().values
    for arity in ("all", "2", "3"):
        mask = (products.split.values == "test") & unique
        if arity != "all":
            mask &= (products.row_3.values >= 0) == (arity == "3")
        for j, name in enumerate(names):
            a, b = y[mask, j], pred[mask, j]
            rho = float(spearmanr(a, b).statistic) if np.std(a) and np.std(b) else None
            result.append(dict(method=method, endpoint=name, arity=arity, n=len(a),
                               spearman=rho, mae=float(np.abs(a-b).mean()), rmse=float(np.sqrt(np.mean((a-b)**2)))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/comparison"))
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", default=38026, type=int)
    p.add_argument("--epochs", default=250, type=int)
    p.add_argument("--foundation-root", type=Path, help="Optional existing minimol/surrogate foundation directory")
    p.add_argument("--models-root", type=Path, help="Optional root containing existing per-repository model folders")
    p.add_argument("--descriptors-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(args.out, args.seed)
    products = pd.read_csv(args.out / "products.csv", dtype={"reaction_id": str})
    synthons = pd.read_csv(args.out / "synthons.csv", dtype={"reaction_id": str})
    bp = np.load(args.out / "basis.npz")
    unique, inv = np.unique(products.smiles, return_inverse=True)
    rows = products[[f"row_{j}" for j in (1, 2, 3)]].values
    split = products.split.values
    repositories = ["rdkit"] + ([] if args.descriptors_only else list(REVISIONS))
    metrics, all_y, all_names, all_predictions, timings = [], [], [], {}, {}
    for repository in repositories:
        print(f"Starting {repository}", flush=True)
        cache = args.out / f"{repository}.npz"
        start = time.perf_counter()
        if not cache.exists():
            if repository == "rdkit":
                names = [f"RDKit/{s}" for s in DESCRIPTORS]
                y = np.array([[f(Chem.MolFromSmiles(s)) for f in DESCRIPTORS.values()] for s in unique])[inv]
                x = RDKitTeacher().encode(synthons.capped_smiles)
                provenance = {"teacher": "RDKit descriptors"}
            else:
                teacher = OpenADMETTeacher(repository, device=args.device, batch_size=128,
                                          foundation_root=args.foundation_root,
                                          model_dir=args.models_root / repository if args.models_root else None)
                if any(":output_" in t for t in teacher.targets):
                    # Local deployment bundles may omit recipe YAML. Recover
                    # target names only from our independently verified model
                    # manifest, and require the exact checkpoint hash to match.
                    manifest = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/adapter_smoke.json").read_text())
                    verified = next(m for m in manifest if m["repository"] == repository)
                    if verified["checkpoint_sha256"] != teacher.provenance["checkpoint_sha256"]:
                        raise ValueError("Local checkpoint has no verified target order")
                    teacher.targets = verified["targets"]
                    teacher.provenance["target_order_source"] = "hash-matched benchmarks/adapter_smoke.json"
                names = [f"{repository}/{t}" for t in teacher.targets]
                y = teacher.predict(unique.tolist())[inv]
                x = teacher.encode(synthons.capped_smiles.tolist())
                provenance = teacher.provenance
                del teacher; gc.collect(); torch.cuda.empty_cache()
            np.savez_compressed(cache, x=x, y=y, names=names)
            dump(args.out / f"{repository}_teacher.json", {**provenance, "load_label_encode_seconds": time.perf_counter()-start})
        data = np.load(cache)
        x, y, names = data["x"], data["y"], data["names"].tolist()
        predictions = {"shi_eq11": shi_predict(y, bp["basis"], bp["refs"])}
        # Additional train-only affine calibration, per endpoint and reaction.
        calibrated = np.zeros_like(y)
        for rid in products.reaction_id.unique():
            selected = products.reaction_id.values == rid
            tr = selected & (split == "train")
            for j in range(y.shape[1]):
                design = np.column_stack([predictions["shi_eq11"][:, j], np.ones(len(y))])
                coef = np.linalg.lstsq(design[tr], y[tr, j], rcond=None)[0]
                calibrated[selected, j] = design[selected] @ coef
        predictions["shi_calibrated"] = calibrated
        predictions["learned_additive"], addinfo = additive_fit(y, rows, split)
        cpfile = args.out / f"{repository}_cp.npz"
        if not cpfile.exists():
            start = time.perf_counter()
            cppred, info, state = fit_cp(x, y, rows, split, args.seed, args.device, args.epochs)
            info["fit_seconds"] = time.perf_counter() - start
            np.savez_compressed(cpfile, predictions=cppred)
            torch.save(state, args.out / f"{repository}_cp.pt")
            dump(args.out / f"{repository}_cp.json", info)
        predictions["cp16"] = np.load(cpfile)["predictions"]
        timings[repository] = {"cp": json.loads((args.out / f"{repository}_cp.json").read_text()), "additive": addinfo}
        for method, pred in predictions.items():
            metrics.extend(metric_rows(y, pred, names, method, products))
            all_predictions.setdefault(method, []).append(pred)
        all_y.append(y); all_names.extend(names)
        print(f"Finished {repository}", flush=True)
        pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
    np.savez_compressed(args.out / "comparison.npz", y=np.concatenate(all_y, axis=1), names=all_names,
                        **{k: np.concatenate(v, axis=1) for k, v in all_predictions.items()})
    dump(args.out / "training.json", timings)
    print(f"Saved {args.out}/metrics.csv", flush=True)


if __name__ == "__main__":
    main()
