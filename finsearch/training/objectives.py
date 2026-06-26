"""Triplet-loss training objectives.

Note: :class:`TripletMarginLoss` here is FinSearch's own implementation (cosine
distance by default) and is distinct from ``torch.nn.TripletMarginLoss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DISTANCE_METRICS = ("cosine", "euclidean")


def pairwise_row_distance(x1: torch.Tensor, x2: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Row-wise distance between two ``(batch, dim)`` tensors.

    ``cosine`` returns ``1 - cosine_similarity``; ``euclidean`` returns the L2 norm.
    """
    if metric == "cosine":
        return 1 - (F.normalize(x1, p=2, dim=1) * F.normalize(x2, p=2, dim=1)).sum(dim=1)
    if metric == "euclidean":
        return torch.norm(x1 - x2, p=2, dim=1)
    raise ValueError(f"Unknown distance metric '{metric}'. Choose one of {DISTANCE_METRICS}.")


class TripletMarginLoss(nn.Module):
    """``max(0, d(anchor, positive) - d(anchor, negative) + margin)``.

    Args:
        margin: Required gap between positive and negative distances.
        distance_metric: ``"cosine"`` or ``"euclidean"``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(self, margin: float = 0.5, distance_metric: str = "cosine", reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        positive_distance = pairwise_row_distance(
            anchor_embeddings, positive_embeddings, self.distance_metric
        )
        negative_distance = pairwise_row_distance(
            anchor_embeddings, negative_embeddings, self.distance_metric
        )
        losses = F.relu(positive_distance - negative_distance + self.margin)

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses

    def __repr__(self) -> str:
        return (
            f"TripletMarginLoss(margin={self.margin}, "
            f"distance_metric='{self.distance_metric}', reduction='{self.reduction}')"
        )


class OnlineTripletLoss(nn.Module):
    """Triplet loss with in-batch mining over ``(embeddings, labels)``.

    Not used by the default training scripts; kept as an alternative objective.

    Args:
        margin: Required gap between positive and negative distances.
        mining_strategy: ``"hard"`` (hardest positive / hardest negative per anchor)
            or ``"all"`` (average over every valid triplet).
        distance_metric: ``"cosine"`` or ``"euclidean"``.
    """

    def __init__(self, margin: float = 0.5, mining_strategy: str = "hard", distance_metric: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance_metric = distance_metric

    def _pairwise_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "euclidean":
            dot_product = torch.mm(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            squared = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            return torch.sqrt(F.relu(squared) + 1e-8)
        normalized = F.normalize(embeddings, p=2, dim=1)
        return 1 - torch.mm(normalized, normalized.t())

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = self._pairwise_distance_matrix(embeddings)
        labels = labels.view(-1, 1)

        positive_mask = (labels == labels.t()).float()
        positive_mask.fill_diagonal_(0)
        negative_mask = (labels != labels.t()).float()

        if self.mining_strategy == "hard":
            hardest_positive, _ = (distances * positive_mask).max(dim=1)
            hardest_negative, _ = (distances + (1 - negative_mask) * 1e9).min(dim=1)
            losses = F.relu(hardest_positive - hardest_negative + self.margin)

            valid_anchors = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
            losses = losses * valid_anchors.float()
            return losses.sum() / valid_anchors.sum().clamp(min=1)

        if self.mining_strategy == "all":
            triplet_losses = F.relu(distances.unsqueeze(2) - distances.unsqueeze(1) + self.margin)
            valid_mask = positive_mask.unsqueeze(2) * negative_mask.unsqueeze(1)
            return (triplet_losses * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        raise ValueError(f"Unknown mining strategy '{self.mining_strategy}'. Choose 'hard' or 'all'.")
