"""Scan a reaction's Cartesian space using cached factors; keep only top-k."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chemistry import slot_groups
from model import cartesian_blocks


@torch.inference_mode()
def top_hits(slots, alpha, beta, k, minimize=False, block=2048):
    """Bounded-memory exact top-k over the APPROXIMATE factor scores."""
    best_values, best_ids = None, None
    for start, stop, scores in cartesian_blocks(slots, alpha, beta, block):
        values, indices = torch.topk(
            scores.flatten(), min(k, scores.numel()), largest=not minimize
        )
        left, right = (
            start + indices // scores.shape[1],
            stop + indices % scores.shape[1],
        )
        ids = torch.stack([left, right], dim=1)
        if best_values is not None:
            values, ids = torch.cat([best_values, values]), torch.cat([best_ids, ids])
        best_values, order = torch.topk(
            values, min(k, len(values)), largest=not minimize
        )
        best_ids = ids[order]
    ids = best_ids.cpu().numpy()
    if len(slots) == 3:
        ids = np.column_stack(
            [ids[:, 0] // len(slots[1]), ids[:, 0] % len(slots[1]), ids[:, 1]]
        )
    return best_values.cpu().numpy(), ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("runs/rdkit/factors.npz"))
    p.add_argument(
        "--reaction",
        required=True,
        help="Exactly one reaction; never mix incompatible slots",
    )
    p.add_argument("--target", type=int, default=0, help="Zero-based endpoint index")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", type=Path, default=Path("runs/hits.csv"))
    args = p.parse_args()
    if args.top_k < 1:
        p.error("top-k must be positive")
    torch.set_num_threads(4)
    cache = np.load(args.cache, allow_pickle=False)
    frame = pd.DataFrame(
        {k: cache[k] for k in ("reaction_id", "position", "synthon_id")}
    )
    groups = slot_groups(frame)
    rows = groups[args.reaction]
    slots = [
        torch.tensor(
            cache["factors"][ids, args.target], device=args.device, dtype=torch.float32
        )
        for ids in rows
    ]
    alpha, beta = float(cache["alpha"][args.target]), float(cache["beta"][args.target])
    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.synchronize()
    start = time.perf_counter()
    values, hits = top_hits(slots, alpha, beta, args.top_k, args.minimize, args.block)
    elapsed = time.perf_counter() - start
    output = []
    for score, ids in zip(values, hits):
        source_rows = [int(rows[s][i]) for s, i in enumerate(ids)]
        output.append(
            {
                "reaction_id": args.reaction,
                "target": str(cache["targets"][args.target]),
                "cp16_score": float(score),
                "row_indices": json.dumps(source_rows),
                "synthon_ids": json.dumps(frame.iloc[source_rows].synthon_id.tolist()),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(args.out, index=False)
    print(
        f"Scored {int(np.prod([len(s) for s in slots])):,} tuples in {elapsed:.3f} s; wrote {args.out}"
    )


if __name__ == "__main__":
    main()
