"""Budgeted docking regression with neural CP factors cached per synthon.

Only acquired utilities enter fitting. Feature normalization uses synthons in
the fitting partition; validation uses a deterministic 20% of acquired rows.
Inference runs gathered factor products, without molecular graph evaluation or
per-product MLP calls. All supplied targets are higher-is-better utilities.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn


class _FactorNetwork(nn.Module):
    def __init__(self, input_dim, n_reactions, rank, hidden, additive):
        super().__init__()
        self.rank, self.additive = rank, additive
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, rank) for _ in range(3)])
        # A third factor near one gives di-/trisynthons comparable initial scale.
        nn.init.ones_(self.heads[2].bias)
        self.add_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)]) if additive else None
        self.reaction_bias = nn.Embedding(n_reactions, 1)
        nn.init.zeros_(self.reaction_bias.weight)
        self.bias = nn.Parameter(torch.zeros(()))

    def components(self, x):
        h = self.trunk(x)
        factors = torch.stack([head(h) for head in self.heads])
        additive = torch.stack([head(h).squeeze(-1) for head in self.add_heads]) if self.additive else None
        return factors, additive

    def combine(self, factors, additive, rows, reaction_ids):
        values = factors[0, rows[:, 0]] * factors[1, rows[:, 1]]
        third = factors[2, rows[:, 2].clamp_min(0)]
        values = values * torch.where(rows[:, 2, None] >= 0, third, 1.0)
        pred = values.sum(-1) / math.sqrt(self.rank)
        if additive is not None:
            pred = pred + additive[0, rows[:, 0]] + additive[1, rows[:, 1]]
            pred = pred + torch.where(rows[:, 2] >= 0, additive[2, rows[:, 2].clamp_min(0)], 0.0)
        return pred + self.reaction_bias(reaction_ids).squeeze(-1) + self.bias

    def forward(self, x, rows, reaction_ids):
        # Encode each distinct synthon once per minibatch, including repeats.
        unique, inverse = torch.unique(rows.clamp_min(0), sorted=False, return_inverse=True)
        factors, additive = self.components(x[unique])
        local_rows = torch.where(rows >= 0, inverse.reshape_as(rows), -1)
        return self.combine(factors, additive, local_rows, reaction_ids)


class FactorRegressor:
    """Small neural CP model with optional additive terms and reaction offsets."""

    def __init__(self, features, n_reactions, rank=16, hidden=128,
                 additive=False, device="cuda", seed=0):
        self.features = np.asarray(features, dtype=np.float32)
        if self.features.ndim != 2 or not np.isfinite(self.features).all():
            raise ValueError("features must be a finite [synthons, features] array")
        if min(n_reactions, rank, hidden) < 1:
            raise ValueError("n_reactions, rank and hidden must be positive")
        self.n_reactions, self.rank, self.hidden = int(n_reactions), int(rank), int(hidden)
        self.additive, self.device, self.seed = bool(additive), torch.device(device), int(seed)
        self.model = None

    def _inputs(self, tuples, reaction_ids):
        rows = np.asarray(tuples, dtype=np.int64)
        reactions = np.asarray(reaction_ids, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != 3 or reactions.shape != (len(rows),):
            raise ValueError("Need tuples [n,3] and reaction_ids [n]")
        if len(rows) and (np.any(rows[:, :2] < 0) or rows.min() < -1
                          or rows.max() >= len(self.features) or reactions.min() < 0
                          or reactions.max() >= self.n_reactions):
            raise ValueError("Invalid synthon or reaction index; only slot 3 may be absent")
        return rows, reactions

    def fit(self, tuples, reaction_ids, y, epochs=120, patience=15, batch_size=512):
        rows, reactions = self._inputs(tuples, reaction_ids)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (len(rows),) or len(y) < 10 or not np.isfinite(y).all():
            raise ValueError("Need at least ten finite acquired utilities")
        if min(epochs, patience, batch_size) < 1:
            raise ValueError("epochs, patience and batch_size must be positive")
        started = time.perf_counter()
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed + 7919)
        order = rng.permutation(len(y))
        cut = max(2, int(len(y) * .2))
        val, train = order[:cut], order[cut:]
        used = np.unique(rows[train][rows[train] >= 0])
        self.x_mean = self.features[used].mean(0)
        self.x_scale = np.maximum(self.features[used].std(0), 1e-3)
        self.x = torch.tensor(np.clip((self.features - self.x_mean) / self.x_scale, -8, 8), device=self.device)
        self.y_mean = float(y[train].mean())
        self.y_scale = max(float(y[train].std()), 1e-5)
        self.model = _FactorNetwork(self.x.shape[1], self.n_reactions, self.rank, self.hidden, self.additive).to(self.device)
        rr = torch.tensor(rows, device=self.device)
        ri = torch.tensor(reactions, device=self.device)
        target = torch.tensor((y - self.y_mean) / self.y_scale, device=self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.002, weight_decay=1e-4)
        best, stale, selected, history = float("inf"), 0, 0, []
        for epoch in range(int(epochs)):
            self.model.train()
            for ids in np.array_split(rng.permutation(train), max(1, math.ceil(len(train) / batch_size))):
                pred = self.model(self.x, rr[ids], ri[ids])
                loss = (pred - target[ids]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite CP training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                error_sum = 0.
                for ids in np.array_split(val, max(1, math.ceil(len(val) / batch_size))):
                    pred = self.model(self.x, rr[ids], ri[ids])
                    error_sum += float((pred - target[ids]).square().sum())
                error = error_sum / len(val)
            history.append(error)
            if error < best:
                best, stale, selected = error, 0, epoch + 1
                state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        self.model.load_state_dict(state)
        self.model.eval()
        self.refresh_cache()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"epochs": epoch + 1, "selected_epoch": selected, "best_val_nmse": best,
                "fit_seconds": time.perf_counter() - started, "train_count": len(train),
                "validation_count": len(val), "validation_history": history,
                "rank": self.rank, "additive": self.additive,
                "normalization": "fitting-partition synthons and acquired fitting-partition utilities"}

    @torch.no_grad()
    def refresh_cache(self, batch_size=4096):
        if self.model is None:
            raise RuntimeError("Call fit first")
        factors, additive = [], []
        for chunk in self.x.split(batch_size):
            f, a = self.model.components(chunk)
            factors.append(f)
            if a is not None:
                additive.append(a)
        self.cached_factors = torch.cat(factors, dim=1)
        self.cached_additive = torch.cat(additive, dim=1) if additive else None

    @torch.no_grad()
    def predict(self, tuples, reaction_ids, batch_size=131072):
        rows, reactions = self._inputs(tuples, reaction_ids)
        if self.model is None:
            raise RuntimeError("Call fit first")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            rr = torch.tensor(rows[start:start + batch_size], device=self.device)
            ri = torch.tensor(reactions[start:start + batch_size], device=self.device)
            pred = self.model.combine(self.cached_factors, self.cached_additive, rr, ri)
            output[start:start + len(rr)] = (pred * self.y_scale + self.y_mean).cpu().numpy()
        return output
