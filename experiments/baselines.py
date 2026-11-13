"""Basis-product additivity and integer histogram convolution controls."""

from collections import Counter
from itertools import product

import numpy as np


def basis_indices(tuples, reaction_ids):
    """Locate the all-reference product and each single-slot substitution.

    Local slot index zero is the reference. Requires complete Cartesian grids.
    Returns basis row indices per product and the reference row index.
    """
    lookup = {(str(r), tuple(x)): i for i, (r, x) in enumerate(zip(reaction_ids, tuples))}
    refs, basis = [], []
    for rid, row in zip(reaction_ids, tuples):
        k = int(np.sum(row >= 0))
        ref = [0] * k + [-1] * (3 - k)
        refs.append(lookup[str(rid), tuple(ref)])
        indices = []
        for j in range(k):
            x = ref.copy()
            x[j] = int(row[j])
            indices.append(lookup[str(rid), tuple(x)])
        basis.append(indices + [-1] * (3 - k))
    return np.asarray(basis), np.asarray(refs)


def shi_predict(y, basis, refs):
    """Shi et al. (2000), Eq. 11; y contains FULL basis-product predictions."""
    present = basis >= 0
    return (y[basis.clip(0)] * present[..., None]).sum(1) - (
        present.sum(1) - 1
    )[:, None] * y[refs]


def integer_histogram(slot_values, offset=0):
    """Exact count convolution of integer additive contributions.

    Python integers prevent count overflow. This is an additive histogram
    control, NOT an implementation of SpaceProp's IPC/EPC boundary chemistry.
    """
    counts = Counter({int(offset): 1})
    for values in slot_values:
        values = np.asarray(values)
        if not np.all(values == np.rint(values)):
            raise ValueError("Histogram control requires integer contributions")
        local = Counter(int(v) for v in values)
        result = Counter()
        for (a, na), (b, nb) in product(counts.items(), local.items()):
            result[a + b] += na * nb
        counts = result
    return counts
