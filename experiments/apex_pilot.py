"""Budget-limited ADMET adaptation using the actual upstream APEX modules.

Keeps the native 64-dimensional, four-layer architectures, LOO ridge probe,
noisy molecular-encoder training and normalized embedding reconstruction loss.
Uses this pilot's split and a short training schedule, NOT the published trained
APEX weights, original docking benchmark, or CUDA top-k extension.
"""

import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .compare import dump, metric_rows
from .prepare_apex import REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/comparison"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", default=1500, type=int)
    p.add_argument("--seed", default=38026, type=int)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Run prepare_apex with the pinned revision")
    sys.path.insert(0, str(args.upstream.resolve()))
    from apex.data.torch_graph import TorchMol, PackedTorchMol
    from apex.data.atom_bond_features import Vocabulary
    from apex.nn.encoder import LigandEncoder
    from apex.nn.probe import LinearProbe
    from apex.nn.apex import APEXFactorizer, APEXFactorizedCSL
    from apex.utils.torch_utils import build_library_indexes
    Vocabulary.register_atom_vocab(None)
    Vocabulary.register_bond_vocab(None)

    class PilotLibrary(APEXFactorizedCSL):
        """Native library encoding with explicit pilot indexes, no file loader."""
        def __init__(self, factorizer):
            nn.Module.__init__(self)
            self.factorizer = factorizer
            self.embed_dim = factorizer.embed_dim
            self.associative_dim = factorizer.associative_dim

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    products = pd.read_csv(args.out / "products.csv", dtype={"reaction_id": str})
    synthons = pd.read_csv(args.out / "synthons.csv", dtype={"reaction_id": str})
    data = np.load(args.out / "comparison.npz")
    y, names = data["y"], data["names"].tolist()
    train = np.flatnonzero(products.split.values == "train")
    val = np.flatnonzero(products.split.values == "validation")
    mean, scale = y[train].mean(0), np.maximum(y[train].std(0), 1e-5)
    target = torch.tensor((y - mean) / scale, dtype=torch.float32, device=device)
    params = dict(node_dim=64, edge_dim=64, graph_dim=64, hidden_dim=64, num_layers=4)
    encoder = LigandEncoder(64, params).to(device)
    probe = LinearProbe(64, names).to(device)
    print("Preparing native APEX molecular graphs", flush=True)
    graphs = [TorchMol.from_smiles(s) for s in products.smiles]

    def graphbatch(ids):
        return PackedTorchMol([graphs[int(i)] for i in ids]).to(device)

    @torch.no_grad()
    def encode(ids):
        return torch.cat([encoder(graphbatch(ii)) for ii in np.array_split(ids, max(1, int(np.ceil(len(ids)/200))))])

    start = time.perf_counter()
    surrogate_path = args.out / "apex_surrogate.pt"
    if not surrogate_path.exists():
        opt = torch.optim.Adam(encoder.parameters(), lr=.001)
        best, history = float("inf"), []
        for step in range(1, args.steps + 1):
            ids = rng.choice(train, 200, replace=False)
            encoder.train(); probe.train()
            embeds = encoder(graphbatch(ids))
            pred = probe(embeds + torch.randn_like(embeds), target[ids])
            loss = ((target[ids] - pred).square().mean(0) / target[ids].var(0, unbiased=False).clamp_min(1e-8)).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite native APEX surrogate loss at {step}")
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0 or step == args.steps:
                encoder.eval(); probe.eval()
                with torch.no_grad():
                    error = (probe(encode(val)) - target[val]).square().mean().item()
                history.append({"step": step, "validation_nmse": error})
                print(f"APEX surrogate {step}: validation NMSE {error:.4f}", flush=True)
                if error < best:
                    best = error
                    state = {"encoder": copy.deepcopy(encoder.state_dict()), "probe": copy.deepcopy(probe.state_dict()), "step": step}
        torch.save(state, surrogate_path)
        dump(args.out / "apex_surrogate_training.json", {"seconds": time.perf_counter()-start, "history": history, "selected_step": state["step"]})
    state = torch.load(surrogate_path, map_location=device, weights_only=True)
    encoder.load_state_dict(state["encoder"]); probe.load_state_dict(state["probe"])
    encoder.eval(); probe.eval()
    latent = encode(np.arange(len(products)))
    with torch.no_grad():
        full_pred = probe(latent).cpu().numpy() * scale + mean
    factorizer = APEXFactorizer(64, 64, 64, 4, params).to(device)
    library = PilotLibrary(factorizer)
    groups = []
    for rid, group in synthons.groupby("reaction_id", sort=True):
        slots = [group.index[group.position == j].tolist() for j in sorted(group.position.unique())]
        indexes = build_library_indexes([slots], device=device)
        orig = indexes["orig_synthon_ids"].cpu().numpy()
        pack = PackedTorchMol([TorchMol.from_smiles(synthons.iloc[i].smiles) for i in orig]).to(device)
        # Each row is unique to its reaction/slot. Associate original rows with
        # the native synthon2rgroup output order, rather than assuming an order.
        mapping = {int(orig[int(s)]): i for i, s in enumerate(indexes["synthon2rgroup"][0])}
        ids = np.flatnonzero(products.reaction_id.values == rid)
        rr = products.iloc[ids][[f"row_{j}" for j in (1, 2, 3)]].values
        present = torch.tensor(rr >= 0, device=device)
        mapped = torch.tensor([[mapping[int(r)] if r >= 0 else 0 for r in row] for row in rr], device=device)
        tr = np.flatnonzero(products.iloc[ids].split.values == "train")
        groups.append(dict(ids=ids, pack=pack, indexes=indexes, mapped=mapped, present=present, train=tr))

    def factors_for(g):
        feats = factorizer.synthon_graph_encoder(g["pack"])
        return library.encode_library(feats, g["indexes"])["synthon_associative_embeds"]

    def combined(factors, g, local_ids):
        return (factors[g["mapped"][local_ids]] * g["present"][local_ids, :, None]).sum(1)

    @torch.no_grad()
    def reconstruct():
        pred = torch.zeros_like(latent)
        for g in groups:
            pred[g["ids"]] = combined(factors_for(g), g, np.arange(len(g["ids"])))
        return pred

    start = time.perf_counter()
    factor_path = args.out / "apex_factorizer.pt"
    if not factor_path.exists():
        opt = torch.optim.Adam(factorizer.parameters(), lr=.001)
        best, history = float("inf"), []
        for step in range(1, args.steps + 1):
            factorizer.train()
            g = groups[int(rng.integers(len(groups)))]
            local = rng.choice(g["train"], min(300, len(g["train"])), replace=False)
            embeds = latent[g["ids"][local]]
            pred = combined(factors_for(g), g, local)
            loss = ((pred-embeds).square() / embeds.var(0, unbiased=False, keepdim=True).clamp_min(1e-8)).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite native APEX factorizer loss at {step}")
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0 or step == args.steps:
                factorizer.eval()
                with torch.no_grad():
                    pred = reconstruct()
                    error = ((pred[val]-latent[val]).square() / latent[train].var(0, unbiased=False).clamp_min(1e-8)).mean().item()
                history.append({"step": step, "validation_embedding_nmse": error})
                print(f"APEX factorizer {step}: validation embedding NMSE {error:.4f}", flush=True)
                if error < best:
                    best = error
                    state = {"factorizer": copy.deepcopy(factorizer.state_dict()), "step": step}
        torch.save(state, factor_path)
        dump(args.out / "apex_factorizer_training.json", {"seconds": time.perf_counter()-start, "history": history, "selected_step": state["step"]})
    state = torch.load(factor_path, map_location=device, weights_only=True)
    factorizer.load_state_dict(state["factorizer"]); factorizer.eval()
    with torch.no_grad():
        factor_pred = probe(reconstruct()).cpu().numpy() * scale + mean
    np.savez_compressed(args.out / "apex_predictions.npz", apex=factor_pred, molecular_surrogate=full_pred)
    rows = metric_rows(y, full_pred, names, "apex_molecular_surrogate", products)
    rows += metric_rows(y, factor_pred, names, "apex_pilot", products)
    pd.DataFrame(rows).to_csv(args.out / "apex_metrics.csv", index=False)
    dump(args.out / "apex_provenance.json", {
        "upstream": "https://github.com/NumerionLabs/apex", "revision": revision,
        "license": "CC BY-NC 4.0; upstream code not redistributed",
        "source_patch": "lazy apex_topk import only", "seed": args.seed,
        "steps_per_stage": args.steps, "device": device,
        "adaptation": "same 64-dimensional native modules; ADMET+descriptor labels; pilot split; 1500-step default vs upstream 100000; batch variance clamped at 1e-8; factorizer checkpoint selected by validation embedding NMSE",
        "topk_extension_run": False,
    })
    print("Saved native-module APEX pilot", flush=True)


if __name__ == "__main__":
    main()
