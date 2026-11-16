"""Budget-limited iterative SynFactor search on frozen Freedom benchmarks.

Only BudgetOracle.query exposes labels. Screen all unqueried candidates with
cached CP factors; optionally rerank its top fraction and diversify acquisition.
"""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .factor_model import FactorRegressor


class BudgetOracle:
    def __init__(self, cache, target, budget, objective, kind):
        self.__truth = np.load(cache / f"oracle_{target}.npy", mmap_mode="r")
        self.budget, self.objective, self.kind = budget, objective, kind
        self.observed, self.values = [], []
        self.seen = set()

    def query(self, indices):
        indices = [int(i) for i in indices]
        if any(i < 0 or i >= len(self.__truth) for i in indices):
            raise IndexError("Oracle candidate index outside dataset")
        if len(set(indices)) != len(indices) or self.seen.intersection(indices):
            raise ValueError("Repeated candidate requested")
        if len(self.observed) + len(indices) > self.budget:
            raise ValueError("Oracle budget exceeded")
        if self.kind == "surrogate":
            values = -self.__truth[indices, 0]
        elif self.objective == "raw":
            values = -self.__truth[indices, 1]
        else:
            values = self.__truth[indices, 0]
        values = np.where(np.isfinite(values), values, 0).astype(np.float32)
        self.observed.extend(indices); self.values.extend(values.tolist()); self.seen.update(indices)
        return values


def warm_start(cache, manifest, count, seed):
    """Historical GA-DCSO first-batch sampling: uniform reaction, then slot."""
    frame = pd.read_csv(cache / "synthons.csv")
    layout = manifest["layout"]
    total = max(x["offset"] + int(np.prod([len(s) for s in x["slots"]])) for x in layout.values())
    inverse = np.full(total, -1, np.int64)
    enumeration = np.load(cache / "enum_indices.npy", mmap_mode="r")
    if count < 0 or count > len(enumeration):
        raise ValueError("Warm-start count exceeds available candidates")
    inverse[enumeration] = np.arange(len(enumeration))
    rng = np.random.default_rng(seed + 104729)
    reactions = manifest["reactions"]
    axes = {}
    identities = {}
    for rid in reactions:
        axes[rid] = [np.argsort(frame.iloc[ss].synthon_id.values, kind="stable") for ss in layout[rid]["slots"]]
        identities[rid] = [frame.iloc[ss].synthon_id.values for ss in layout[rid]["slots"]]
    found, seen = [], set()
    while len(found) < count:
        rid = reactions[int(rng.integers(len(reactions)))]
        index, chosen = 0, []
        for ss in axes[rid]:
            local = int(rng.choice(ss)); chosen.append(local)
            index = index * len(ss) + local
        row = int(inverse[layout[rid]["offset"] + index])
        if row < 0:
            # Archived GA sees ID tuples, including synonymous attachment rows.
            # Resolve aliases to the same frozen product ID rather than drawing
            # a new random tuple (which would silently change the warm start).
            alternatives = [np.flatnonzero(ids == ids[local]) for ids, local in zip(identities[rid], chosen)]
            for alias in itertools.product(*alternatives):
                other = 0
                for local, axis in zip(alias, axes[rid]):
                    other = other * len(axis) + int(local)
                row = int(inverse[layout[rid]["offset"] + other])
                if row >= 0:
                    break
        if row >= 0 and row not in seen:
            found.append(row); seen.add(row)
    return np.array(found)


def diverse_select(ids, scores, count, packed, bitorder, references, strength, device):
    """Soft Tanimoto penalty against batch and acquired elites, label-free."""
    if strength <= 0:
        return ids[np.argsort(-scores, kind="stable")[:count]]
    keep = np.argsort(-scores, kind="stable")[:min(len(ids), max(5000, count * 8))]
    ids, scores = ids[keep], scores[keep]
    bits = np.unpackbits(packed[ids], axis=1, bitorder=bitorder).astype(np.float32)
    x = torch.tensor(bits, device=device)
    sizes = x.sum(1)
    novelty = torch.zeros(len(ids), device=device)
    if len(references):
        ref = torch.tensor(np.unpackbits(packed[references], axis=1, bitorder=bitorder).astype(np.float32), device=device)
        intersection = x @ ref.T
        novelty = (intersection / (sizes[:, None] + ref.sum(1)[None, :] - intersection).clamp_min(1)).max(1).values
    utility = torch.tensor(scores, device=device)
    penalty = float(strength) * max(float(np.quantile(scores, .95) - np.quantile(scores, .05)), .05)
    selected, unavailable = [], torch.zeros(len(ids), dtype=torch.bool, device=device)
    for _ in range(min(count, len(ids))):
        value = (utility - penalty * novelty).masked_fill(unavailable, -torch.inf)
        j = int(value.argmax())
        selected.append(j); unavailable[j] = True
        intersection = x @ x[j]
        similarity = intersection / (sizes + sizes[j] - intersection).clamp_min(1)
        novelty = torch.maximum(novelty, similarity)
    return ids[selected]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("runs/docking_data/1M"))
    p.add_argument("--target", choices=["kif11", "pyrd", "tgfr1"], default="kif11")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--additive", action="store_true")
    p.add_argument("--rerank", choices=["none", "cp64", "gbm", "cp64-gbm"], default="none")
    p.add_argument("--screen-fraction", type=float, default=.1)
    p.add_argument("--diversity", type=float, default=0.)
    p.add_argument("--explore", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--disagreement", type=float, default=0.)
    p.add_argument("--features", default="features.npy")
    p.add_argument("--objective", choices=["stored", "raw"], default="stored")
    p.add_argument("--chemical-filter", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--max-seconds", type=float, default=1800)
    args = p.parse_args()
    if not 0 <= args.explore < 1 or not 0 < args.screen_fraction <= 1:
        p.error("Invalid exploration/screen fraction")
    if args.ensemble < 1 or args.budget < 10 or args.batch < 10:
        p.error("Need positive ensemble and at least ten initial/total queries")
    if (args.out / "observed.csv").exists():
        raise ValueError("Use a fresh output directory; refusing to overwrite a run")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(scale=manifest["scale"], oracle_kind=manifest["kind"],
                  objective_definition="stored glide_le = -raw Glide - MolLogP, zero on failure/filter" if manifest["kind"]=="glide" and args.objective=="stored" else "negative docking score",
                  acquisitions="predictions only; oracle labels accessed through charged query()",
                  fingerprint_type=manifest["fingerprint_type"])
    (args.out / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    rows = np.load(args.cache / "tuples.npy", mmap_mode="r")
    rids = np.load(args.cache / "reaction_ids.npy", mmap_mode="r")
    features = np.load(args.cache / args.features)
    pids = np.load(args.cache / "product_ids.npy", mmap_mode="r")
    packed = np.load(args.cache / "fingerprints.npy", mmap_mode="r")
    eligible = np.ones(len(rows), bool)
    if args.chemical_filter:
        eligible &= np.load(args.cache / "chemistry_filter.npy")
    queried = np.zeros(len(rows), bool)
    oracle = BudgetOracle(args.cache, args.target, args.budget, args.objective, manifest["kind"])
    rng = np.random.default_rng(args.seed + 991)
    started = time.perf_counter()
    initial = warm_start(args.cache, manifest, min(args.batch, args.budget), args.seed)
    oracle.query(initial); queried[initial] = True
    history = []

    def save():
        ids = np.asarray(oracle.observed)
        pd.DataFrame({"call": np.arange(1, len(ids)+1), "row": ids, "product_id": pids[ids].astype(str),
                      "utility": oracle.values}).to_csv(args.out / "observed.csv", index=False)
        (args.out / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    save()
    while len(oracle.observed) < args.budget and time.perf_counter()-started < args.max_seconds:
        iteration = len(history) + 1
        train = np.asarray(oracle.observed)
        y = np.asarray(oracle.values, np.float32)
        t0 = time.perf_counter()
        members, fit = [], []
        for member in range(args.ensemble):
            model = FactorRegressor(features, len(manifest["reactions"]), rank=args.rank,
                                    additive=args.additive, device=args.device, seed=args.seed + iteration*17 + member*101)
            fit.append(model.fit(rows[train], rids[train], y, epochs=args.epochs))
            members.append(model.predict(rows, rids))
            del model
        fit_done = time.perf_counter()
        predictions = np.mean(members, axis=0)
        disagreement = np.std(members, axis=0)
        predictions += args.disagreement * disagreement
        predictions[queried | ~eligible] = -np.inf
        available = int((~queried & eligible).sum())
        count = min(args.batch, args.budget - len(train), available)
        if count == 0:
            break
        n_explore = min(int(round(count * args.explore)), count-1)
        pool_size = min(int((~queried & eligible).sum()), max(count*10, int(len(rows)*args.screen_fraction)))
        pool = np.argpartition(predictions, -pool_size)[-pool_size:]
        scores = predictions[pool].copy()
        refit = None
        if args.rerank in ("cp64", "cp64-gbm"):
            fine = FactorRegressor(features, len(manifest["reactions"]), rank=64, additive=True,
                                   device=args.device, seed=args.seed + iteration*17 + 3)
            refit = fine.fit(rows[train], rids[train], y, epochs=args.epochs)
            scores = fine.predict(rows[pool], rids[pool])
            del fine
        if args.rerank in ("gbm", "cp64-gbm"):
            from lightgbm import LGBMRegressor
            regressor = LGBMRegressor(n_estimators=250, num_leaves=31, learning_rate=.05,
                                     min_child_samples=20, colsample_bytree=.8, reg_lambda=1.,
                                     n_jobs=4, verbosity=-1, random_state=args.seed+iteration)
            xtrain = np.unpackbits(packed[train], axis=1, bitorder=manifest["bitorder"])
            regressor.fit(xtrain, y)
            # Both predictions are on the same observed-oracle utility scale.
            reranked = np.empty(len(pool), np.float32)
            for start in range(0, len(pool), 20000):
                ix = pool[start:start+20000]
                xx = np.unpackbits(packed[ix], axis=1, bitorder=manifest["bitorder"])
                reranked[start:start+len(ix)] = regressor.predict(xx)
            scores = .2 * scores + .8 * reranked
            del regressor, xtrain
        if args.rerank != "none":
            scores += args.disagreement * disagreement[pool]
        references = train[np.argsort(-y, kind="stable")[:100]]
        chosen = diverse_select(pool, scores, count-n_explore, packed, manifest["bitorder"], references,
                                args.diversity, args.device)
        remaining = ~queried & eligible
        remaining[chosen] = False
        random = rng.choice(np.flatnonzero(remaining), size=n_explore, replace=False)
        acquired = np.concatenate([chosen, random])
        got = oracle.query(acquired); queried[acquired] = True
        elapsed = time.perf_counter() - started
        summary = dict(iteration=iteration, calls=len(oracle.observed), best_utility=max(oracle.values),
                       batch_mean=float(got.mean()), batch_positive=int(np.sum(got>0)),
                       fit_seconds=fit_done-t0, round_seconds=time.perf_counter()-t0, elapsed_seconds=elapsed,
                       fit=fit, fine_fit=refit)
        history.append(summary); save()
        print(f"round={iteration} calls={len(oracle.observed)} best={max(oracle.values):.4f} batch={got.mean():.3f} elapsed={elapsed:.1f}s", flush=True)
        del predictions, members, disagreement; gc.collect()
    result = dict(config=config, calls=len(oracle.observed), unique_calls=len(oracle.seen),
                  wall_seconds=time.perf_counter()-started, completed_budget=len(oracle.observed)==args.budget,
                  best_utility=max(oracle.values), initial_ids_sha256=hashlib.sha256(pids[initial].tobytes()).hexdigest())
    (args.out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
