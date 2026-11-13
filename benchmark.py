"""Reproduce the hERG matrix benchmark; calculate scaling from measured seconds.

Default inputs are the exact frozen factor matrices used for the README timing.
They contain learned numbers only, not structures or the teacher checkpoint.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--factors",
        type=Path,
        default=Path(__file__).resolve().parent
        / "benchmarks/herg_two_slot_factors.npz",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--copy-to-host", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs/benchmark.json"))
    args = p.parse_args()
    if min(args.block, args.repeats) < 1:
        p.error("block and repeats must be positive")
    torch.set_num_threads(4)
    cache = np.load(args.factors, allow_pickle=False)
    a = torch.tensor(cache["A"], device=args.device, dtype=torch.float32)
    b = torch.tensor(cache["B"], device=args.device, dtype=torch.float32)
    alpha, beta = float(cache["alpha"]), float(cache["beta"])
    cuda = args.device == "cuda"
    if cuda:
        torch.set_float32_matmul_precision("high")
    buffers = {}
    for i in range(0, len(a), args.block):
        for j in range(0, len(b), args.block):
            shape = min(args.block, len(a) - i), min(args.block, len(b) - j)
            if shape not in buffers:
                buffers[shape] = (
                    torch.empty(shape, device=args.device),
                    torch.empty(shape, pin_memory=cuda) if args.copy_to_host else None,
                )

    @torch.inference_mode()
    def sweep():
        for i in range(0, len(a), args.block):
            left = a[i : i + args.block]
            for j in range(0, len(b), args.block):
                right = b[j : j + args.block]
                output, host = buffers[len(left), len(right)]
                torch.mm(left, right.T, out=output)
                output.mul_(alpha).add_(beta)
                if host is not None:
                    host.copy_(output, non_blocking=cuda)
                    if cuda:
                        torch.cuda.synchronize()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    sweep()
    sync()
    seconds = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        sweep()
        sync()
        seconds.append(time.perf_counter() - start)
    expected = a[:32].double() @ b[:32].double().T * alpha + beta
    observed = (a[:32] @ b[:32].T) * alpha + beta
    max_error = float((expected - observed).abs().max())
    rate = len(a) * len(b) / statistics.median(seconds)
    result = {
        "device": torch.cuda.get_device_name() if cuda else "CPU",
        "torch": torch.__version__,
        "slot_sizes": [len(a), len(b)],
        "rank": a.shape[1],
        "distinct_tuples_per_sweep": len(a) * len(b),
        "block": args.block,
        "seconds": seconds,
        "median_scores_per_second": rate,
        "copy_to_host": args.copy_to_host,
        "max_error_vs_float64": max_error,
        "extrapolated_seconds": {
            str(n): n / rate for n in (10**9, 10**12, 20 * 10**12)
        },
        "scope": "warm two-slot Cartesian matrix kernel + affine; optional host copy; no training, encoder, assembly, ranking, filtering or disk persistence",
    }
    if not math.isfinite(max_error) or max_error > 1e-4:
        raise RuntimeError(f"Unexpected matrix accuracy: {max_error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
