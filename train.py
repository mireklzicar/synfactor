"""Fit CP16 against product teacher labels; export reusable synthon factors."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from model import SynFactor, score_tuples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("runs/rdkit"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--seed", type=int, default=38026)
    args = p.parse_args()
    if min(args.epochs, args.patience, args.rank) < 1:
        p.error("epochs, patience and rank must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed + args.rank)
    dataset = np.load(args.data / "dataset.npz", allow_pickle=False)
    metadata = json.loads((args.data / "dataset.json").read_text())
    frame = pd.read_csv(
        args.data / "synthons.csv", dtype={"synthon_id": str, "reaction_id": str}
    )
    x = torch.tensor(
        dataset["embeddings"][dataset["row_to_embedding"]], device=args.device
    )
    tuples = torch.tensor(dataset["tuples"], device=args.device)
    present = tuples >= 0
    y = torch.tensor(dataset["targets"], device=args.device)
    fit = torch.tensor(np.flatnonzero(dataset["split"] == "train"), device=args.device)
    val = torch.tensor(
        np.flatnonzero(dataset["split"] == "validation"), device=args.device
    )
    model = SynFactor(x.shape[1], y.shape[1], args.rank).to(args.device)
    with torch.no_grad():
        occurrences = x[tuples[fit][present[fit]]]
        model.x_mean.copy_(occurrences.mean(0))
        std = occurrences.std(0, correction=0)
        model.x_scale.copy_(torch.where(std > 1e-5, std, 1.0))
        model.y_mean.copy_(y[fit].mean(0))
        model.y_scale.copy_(y[fit].std(0).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    best, stale, best_epoch, state = float("inf"), 0, 0, None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for ids in fit[torch.randperm(len(fit), device=args.device)].split(256):
            prediction = model(x[tuples[ids].clamp_min(0)], present[ids])
            loss = torch.nn.functional.huber_loss(
                (prediction - model.y_mean) / model.y_scale,
                (y[ids] - model.y_mean) / model.y_scale,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x[tuples[val].clamp_min(0)], present[val])
            val_loss = (((prediction - y[val]) / model.y_scale) ** 2).mean().item()
        if val_loss < best - 1e-5:
            best, stale, best_epoch = val_loss, 0, epoch + 1
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 25 == 0:
            print(f"Epoch {epoch + 1}: validation MSE={val_loss:.4f}", flush=True)
        if stale >= args.patience:
            break
    if state is None:
        raise RuntimeError("Training did not produce a finite validation loss")
    if args.device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - start
    model.load_state_dict(state)
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        factors = torch.empty(
            (len(x), model.tasks, model.rank), device=args.device, dtype=torch.float16
        )
        for slot in range(3):
            ids = torch.tensor(
                np.flatnonzero(frame.position.to_numpy() == slot + 1),
                device=args.device,
            )
            for block in ids.split(4096):
                if len(block):
                    factors[block] = model.factor(x[block], slot).half()
        alpha, beta = model.affine()
        predicted = score_tuples(factors, tuples, alpha, beta).cpu().numpy()
        direct = model(x[tuples.clamp_min(0)], present).cpu().numpy()
    cache_seconds = time.perf_counter() - start
    # All exported entries are plain tensors/strings, compatible with weights_only loading.
    torch.save(
        {
            "config": {
                "input_dim": x.shape[1],
                "tasks": model.tasks,
                "rank": model.rank,
            },
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "targets": metadata["targets"],
        },
        args.data / "model.pt",
    )
    np.savez_compressed(
        args.data / "factors.npz",
        factors=factors.cpu().numpy(),
        alpha=alpha.cpu().numpy(),
        beta=beta.cpu().numpy(),
        targets=np.asarray(metadata["targets"]),
        reaction_id=frame.reaction_id.to_numpy(str),
        position=frame.position.to_numpy(np.int64),
        synthon_id=frame.synthon_id.to_numpy(str),
    )
    reports = []
    for split in ("train", "validation", "test"):
        keep = dataset["split"] == split
        for t, name in enumerate(metadata["targets"]):
            for arity in ("all", 2, 3):
                mask = (
                    keep
                    if arity == "all"
                    else keep & ((dataset["tuples"] >= 0).sum(1) == arity)
                )
                if mask.sum() < 2:
                    continue
                truth, pred = dataset["targets"][mask, t], predicted[mask, t]
                rho = float(spearmanr(truth, pred).statistic)
                reports.append(
                    {
                        "split": split,
                        "components": arity,
                        "target": name,
                        "n": int(mask.sum()),
                        "spearman": rho if np.isfinite(rho) else None,
                        "mae": float(np.abs(truth - pred).mean()),
                    }
                )
    metrics = {
        "device": args.device,
        "torch": torch.__version__,
        "seed": args.seed,
        "rank": args.rank,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "factor_export_seconds": cache_seconds,
        "max_float16_cache_difference": float(np.max(np.abs(direct - predicted))),
        "factor_payload_bytes": factors.numel() * 2,
        "metrics": reports,
    }
    (args.data / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    products = pd.read_csv(args.data / "products.csv")
    for t, name in enumerate(metadata["targets"]):
        products[f"CP16_{name}"] = predicted[:, t]
    products[dataset["split"] == "test"].to_csv(
        args.data / "test_predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
