"""Time a loaded OpenADMET teacher on assembled product SMILES."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from teachers import OpenADMETTeacher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", type=Path, default=Path("runs/herg/products.csv"))
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument("--repository", default="herg-chemeleon-baseline")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--foundation-root", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs/teacher_benchmark.json"))
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    torch.set_num_threads(4)
    smiles = pd.read_csv(args.products)[args.smiles_column]
    start = time.perf_counter()
    teacher = OpenADMETTeacher(
        args.repository, args.device, args.model_dir, args.foundation_root
    )
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    loader, *_ = teacher.featurizer.featurize(smiles)
    prep_seconds = time.perf_counter() - start

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def predict():
        output = []
        for bmg, vertex, extra, *_ in loader:
            bmg.to(args.device)
            vertex = None if vertex is None else vertex.to(args.device)
            extra = None if extra is None else extra.to(args.device)
            latent = teacher.estimator.fingerprint(bmg, vertex, extra)
            output.append(teacher.estimator.predictor(latent).cpu().numpy())
        pred = np.concatenate(output)
        if len(pred) != len(smiles) or not np.isfinite(pred).all():
            raise ValueError("Invalid teacher output")

    predict()
    seconds = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        predict()
        sync()
        seconds.append(time.perf_counter() - start)
    rate = len(smiles) / statistics.median(seconds)
    result = {
        **teacher.provenance,
        "products": len(smiles),
        "seconds": seconds,
        "load_seconds": load_seconds,
        "graph_preparation_seconds": prep_seconds,
        "products_per_second": rate,
        "device": args.device,
        "batch_size": 256,
        "extrapolated_seconds": {
            str(n): n / rate for n in (10**9, 10**12, 20 * 10**12)
        },
        "scope": "warm inference including batch iteration and predictions copied to CPU; initial graph preparation and model loading timed separately",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
