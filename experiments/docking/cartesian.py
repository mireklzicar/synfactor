"""Bounded-memory whole-space top-k scoring from cached neural CP factors.

No enumerated product table, assembly, fingerprints or oracle values are needed.
Two-slot blocks use A @ B.T; three-slot blocks use (A[i] * B[j]) @ C.T.
Ranking is exact for the evaluated float32 GEMM scores. GEMM accumulation can
differ slightly from elementwise ``FactorRegressor.predict`` accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch


@dataclass(frozen=True)
class ReactionGrid:
    reaction_index: int
    slots: tuple


class IncompleteSweepError(TimeoutError):
    """A time-limited partial sweep is not a valid global top-k result."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Cartesian sweep stopped after {diagnostics['visited_products']:,} / "
                         f"{diagnostics['total_products']:,} products")


def _grids(grids, model):
    result, seen = [], set()
    for item in grids:
        grid = item if isinstance(item, ReactionGrid) else ReactionGrid(item["reaction_index"], tuple(item["slots"]))
        rid = int(grid.reaction_index)
        if rid in seen or not 0 <= rid < model.n_reactions:
            raise ValueError("Reaction indices must be distinct and valid")
        seen.add(rid)
        slots = tuple(np.asarray(axis, dtype=np.int64) for axis in grid.slots)
        if len(slots) not in (2, 3):
            raise ValueError("Only two- and three-slot reaction grids are supported")
        for axis in slots:
            if axis.ndim != 1 or not len(axis) or len(np.unique(axis)) != len(axis):
                raise ValueError("Each slot must be a nonempty vector of distinct synthon rows")
            if axis.min() < 0 or axis.max() >= len(model.features):
                raise ValueError("Synthon row outside fitted factor cache")
        result.append(ReactionGrid(rid, slots))
    if not result:
        raise ValueError("Need at least one nonempty reaction grid")
    return sorted(result, key=lambda grid: grid.reaction_index)


def _excluded_by_grid(grids, tuples, reactions):
    if tuples is None and reactions is None:
        return {}
    rows, rids = np.asarray(tuples, np.int64), np.asarray(reactions, np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3 or rids.shape != (len(rows),):
        raise ValueError("Excluded tuples/reaction IDs must have shapes [n,3] and [n]")
    if set(rids.tolist()) - {g.reaction_index for g in grids}:
        raise ValueError("Excluded reaction is absent from the supplied grids")
    result = {}
    for grid in grids:
        selected = rows[rids == grid.reaction_index]
        if not len(selected):
            continue
        maps = [{int(row): local for local, row in enumerate(axis)} for axis in grid.slots]
        encoded = []
        for row in selected:
            if len(maps) == 2 and row[2] != -1:
                raise ValueError("Two-slot excluded route has a nonempty third slot")
            code = 0
            for axis, lookup, value in zip(grid.slots, maps, row):
                if int(value) not in lookup:
                    raise ValueError("Excluded synthon is absent from its ordered reaction slot")
                code = code * len(axis) + lookup[int(value)]
            encoded.append(code)
        result[grid.reaction_index] = np.asarray(sorted(set(encoded)), np.int64)
    return result


def _block_topk(flat, k):
    """Exact float32 ordering with deterministic first-local-index tie breaks.

    Pack the monotone IEEE float32 bit ordering and inverse local index into
    int64. Unlike epsilon perturbations this never reverses unequal scores.
    Blocks contain fewer than 2**32 entries. The extra key arrays are bounded
    by block size, not by space size.
    """
    if len(flat) <= k:
        return flat, torch.arange(len(flat), dtype=torch.int64, device=flat.device)
    bits = flat.view(torch.int32).to(torch.int64)
    # Treat signed zero as the same numerical score before resolving ties.
    bits.masked_fill_(bits == -2147483648, 0)
    ordered = torch.where(bits < 0, -bits - 2147483649, bits)
    local = torch.arange(len(flat), dtype=torch.int64, device=flat.device)
    key = (ordered << 32) | (4294967295 - local)
    indices = torch.topk(key, min(k, len(flat)), sorted=False).indices
    return flat[indices], indices


def _merge(scores, ordinals, added_scores, added_ordinals, k):
    values = torch.cat((scores, added_scores))
    ids = torch.cat((ordinals, added_ordinals))
    # Stable secondary/primary sorts preserve the exact score ordering and
    # break ties by the global reaction/local Cartesian ordinal.
    secondary = torch.argsort(ids, stable=True)
    primary = torch.argsort(values[secondary], descending=True, stable=True)[:k]
    keep = secondary[primary]
    return values[keep], ids[keep]


@torch.no_grad()
def stream_topk(model, grids, k=20000, *, excluded_tuples=None,
                excluded_reaction_ids=None, block_rows=2048, block_cols=4096,
                max_seconds=None):
    """Score every supplied grid and return only the best unqueried routes.

    ``grids`` accepts ReactionGrid objects or dictionaries with
    ``reaction_index`` and ``slots`` (global synthon-row arrays in slot order).
    Exclusions are charged route tuples [n,3], using -1 for absent slot three,
    with corresponding reaction indices. Repeated exclusions are harmless.

    Output is a dict with ``tuples``, ``reaction_ids``, ``scores`` and
    ``diagnostics``. Scores are descending; exact floating ties use increasing
    reaction index, then the supplied slots' local Cartesian enumeration order.
    Output contains min(k, eligible routes) distinct routes. Distinct routes
    can still assemble to the same molecule; molecular deduplication is later.

    The deadline is checked between synchronized blocks. Exceeding it raises
    IncompleteSweepError with diagnostics, never returns a partial top-k.
    Float32 GEMM uses IEEE precision; the previous PyTorch precision setting
    is restored afterward. Working storage scales with block size plus k.
    """
    if model.model is None or not hasattr(model, "cached_factors"):
        raise RuntimeError("Fit the model and cache its factors before scoring")
    if min(k, block_rows, block_cols) < 1 or block_rows * block_cols >= 2**32:
        raise ValueError("Need positive k/block sizes and fewer than 2**32 entries per block")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    started = time.perf_counter()
    grids = _grids(grids, model)
    exclusions = _excluded_by_grid(grids, excluded_tuples, excluded_reaction_ids)
    counts = [math.prod(len(a) for a in grid.slots) for grid in grids]
    total = sum(counts)
    if total >= 2**63:
        raise ValueError("Global Cartesian ordinal exceeds int64 capacity")
    device = model.cached_factors.device
    factors, additive = model.cached_factors, model.cached_additive
    if factors.dtype != torch.float32:
        raise ValueError("Cartesian ranking requires float32 cached factors")
    if not torch.isfinite(factors).all() or (additive is not None and not torch.isfinite(additive).all()):
        raise ValueError("Nonfinite cached factors or additive terms")
    scores = torch.empty(0, dtype=torch.float32, device=device)
    ordinals = torch.empty(0, dtype=torch.int64, device=device)
    visited, blocks, offset = 0, 0, 0
    diagnostics = dict(total_products=total, visited_products=0, blocks=0,
                       excluded_routes=sum(map(len, exclusions.values())), complete=False,
                       block_rows=int(block_rows), block_cols=int(block_cols),
                       precision="float32 GEMM, IEEE", tie_policy="score descending; reaction index then local Cartesian ordinal")

    def checkpoint():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        diagnostics.update(visited_products=visited, blocks=blocks,
                           wall_seconds=time.perf_counter()-started)
        if max_seconds is not None and diagnostics["wall_seconds"] > max_seconds:
            raise IncompleteSweepError(dict(diagnostics))

    previous_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        checkpoint()
        for grid, size in zip(grids, counts):
            axes = [torch.as_tensor(axis, device=device) for axis in grid.slots]
            rid = grid.reaction_index
            n_columns = len(axes[-1])
            n_rows = len(axes[0]) if len(axes) == 2 else len(axes[0]) * len(axes[1])
            right = factors[len(axes)-1, axes[-1]]
            right_add = additive[len(axes)-1, axes[-1]] if additive is not None else None
            bias = model.model.reaction_bias.weight[rid, 0] + model.model.bias
            excluded = exclusions.get(rid, np.empty(0, np.int64))
            for row_start in range(0, n_rows, block_rows):
                row_end = min(n_rows, row_start+block_rows)
                local = torch.arange(row_start, row_end, device=device)
                if len(axes) == 2:
                    first = axes[0][local]
                    left = factors[0, first]
                    left_add = additive[0, first] if additive is not None else None
                else:
                    first, second = axes[0][local // len(axes[1])], axes[1][local % len(axes[1])]
                    left = factors[0, first] * factors[1, second]
                    left_add = additive[0, first] + additive[1, second] if additive is not None else None
                lo, hi = np.searchsorted(excluded, [row_start*n_columns, row_end*n_columns])
                row_excluded = excluded[lo:hi]
                for col_start in range(0, n_columns, block_cols):
                    col_end = min(n_columns, col_start+block_cols)
                    values = (left @ right[col_start:col_end].T) / math.sqrt(model.rank)
                    if additive is not None:
                        values += left_add[:, None]
                        values += right_add[col_start:col_end][None, :]
                    values += bias
                    values *= model.y_scale
                    values += model.y_mean
                    if len(row_excluded):
                        columns = row_excluded % n_columns
                        selected = (columns >= col_start) & (columns < col_end)
                        ex = row_excluded[selected]
                        values[torch.as_tensor(ex // n_columns - row_start, device=device),
                               torch.as_tensor(ex % n_columns - col_start, device=device)] = -torch.inf
                    flat = values.flatten()
                    if len(scores) == k:
                        # Once the retained pool is full, values below its
                        # minimum cannot enter the global result. Most later
                        # blocks contribute few/no entries, avoiding a dense
                        # integer-key top-k operation. Include equality so
                        # deterministic boundary ties remain correct.
                        candidates = torch.nonzero(flat >= scores[-1], as_tuple=False).flatten()
                        v, subset_indices = _block_topk(flat[candidates], k)
                        local_indices = candidates[subset_indices]
                    else:
                        v, local_indices = _block_topk(flat, k)
                    if torch.isnan(v).any() or torch.isposinf(v).any():
                        raise FloatingPointError("Nonfinite Cartesian prediction")
                    valid = torch.isfinite(v)
                    v, local_indices = v[valid], local_indices[valid]
                    global_indices = (row_start + local_indices // (col_end-col_start))*n_columns + col_start + local_indices % (col_end-col_start) + offset
                    if len(v):
                        scores, ordinals = _merge(scores, ordinals, v, global_indices, k)
                    visited += (row_end-row_start)*(col_end-col_start)
                    blocks += 1
                    checkpoint()
            offset += size
        diagnostics["complete"] = True
        out_scores, global_ids = scores.cpu().numpy(), ordinals.cpu().numpy()
        out_rows = np.full((len(global_ids), 3), -1, np.int64)
        out_reactions = np.empty(len(global_ids), np.int64)
        offset = 0
        for grid, count in zip(grids, counts):
            selected = (global_ids >= offset) & (global_ids < offset+count)
            code = global_ids[selected] - offset
            out_reactions[selected] = grid.reaction_index
            for slot in reversed(range(len(grid.slots))):
                out_rows[selected, slot] = grid.slots[slot][code % len(grid.slots[slot])]
                code //= len(grid.slots[slot])
            offset += count
        diagnostics.update(returned_routes=len(out_scores), wall_seconds=time.perf_counter()-started)
        return dict(tuples=out_rows, reaction_ids=out_reactions, scores=out_scores, diagnostics=diagnostics)
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous_precision
