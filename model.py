"""A small neural CP model and scoring directly from its cached factors."""

import math

import torch
from torch import nn


class SynFactor(nn.Module):
    def __init__(self, input_dim, tasks=1, rank=16, hidden=64):
        super().__init__()
        self.tasks, self.rank = tasks, rank
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, tasks * rank) for _ in range(3)])
        self.output_scale = nn.Parameter(torch.ones(tasks))
        self.output_bias = nn.Parameter(torch.zeros(tasks))
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_scale", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(tasks))
        self.register_buffer("y_scale", torch.ones(tasks))

    def factor(self, embeddings, slot):
        x = ((embeddings - self.x_mean) / self.x_scale).clamp(-8, 8)
        return self.heads[slot](self.trunk(x)).reshape(-1, self.tasks, self.rank)

    def forward(self, components, present):
        """components: [batch, 3, embedding_dim]; present: [batch, 3]."""
        interaction = self.factor(components[:, 0], 0) * self.factor(
            components[:, 1], 1
        )
        third = self.factor(components[:, 2], 2)
        interaction = interaction * torch.where(present[:, 2, None, None], third, 1.0)
        normalized = interaction.sum(-1) / math.sqrt(self.rank)
        return (
            normalized * self.output_scale + self.output_bias
        ) * self.y_scale + self.y_mean

    def affine(self):
        return (
            self.output_scale * self.y_scale / math.sqrt(self.rank),
            self.output_bias * self.y_scale + self.y_mean,
        )


def score_tuples(factors, tuples, alpha, beta):
    """factors: [rows, tasks, rank]; tuples: [batch, 3], -1 means no third slot."""
    if tuples.ndim != 2 or tuples.shape[1] != 3:
        raise ValueError("Expected three columns; use -1 for the absent third slot")
    if (
        (tuples[:, :2] < 0).any()
        or (tuples >= len(factors)).any()
        or (tuples < -1).any()
    ):
        raise ValueError("Invalid factor row index")
    product = factors[tuples[:, 0]].float() * factors[tuples[:, 1]].float()
    third = factors[tuples[:, 2].clamp_min(0)].float()
    product *= torch.where(tuples[:, 2, None, None] >= 0, third, 1.0)
    return product.sum(-1) * alpha + beta


def cartesian_blocks(slots, alpha=1.0, beta=0.0, block=2048):
    """Yield (left offset, right offset, scores) without materializing the space.

    Each slot is [number of synthons, rank] for ONE endpoint and ONE reaction.
    Two slots: A @ B.T. Three slots: (A[i] * B[j]) @ C.T. For three slots,
    the left index is flattened i * len(B) + j. Outputs live only until the
    next iteration; copy a block if it must outlive the generator step.
    """
    if len(slots) not in (2, 3) or block < 1 or any(len(s) == 0 for s in slots):
        raise ValueError("Need two or three nonempty slots and a positive block size")
    right = slots[-1].float()
    left_count = len(slots[0]) * (len(slots[1]) if len(slots) == 3 else 1)
    for start in range(0, left_count, block):
        if len(slots) == 2:
            left = slots[0][start : start + block].float()
        else:
            ids = torch.arange(
                start, min(start + block, left_count), device=right.device
            )
            left = (
                slots[0][ids // len(slots[1])].float()
                * slots[1][ids % len(slots[1])].float()
            )
        for stop in range(0, len(right), block):
            scores = left @ right[stop : stop + block].T
            scores.mul_(alpha).add_(beta)
            yield start, stop, scores
