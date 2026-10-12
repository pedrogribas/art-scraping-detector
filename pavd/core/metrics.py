from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np


def top1(ranks: Sequence[int | None]) -> float:
    valid = [rank for rank in ranks if rank is not None]
    return sum(rank == 1 for rank in valid) / len(valid) if valid else 0.0


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    valid = [rank for rank in ranks if rank is not None]
    return sum(rank <= k for rank in valid) / len(valid) if valid else 0.0


def mrr(ranks: Sequence[int | None]) -> float:
    valid = [rank for rank in ranks if rank is not None]
    return sum(1.0 / rank for rank in valid) / len(valid) if valid else 0.0


def ann_recall_at_k(true_ids: Iterable[str], retrieved_ids: Iterable[Sequence[str]], k: int) -> float:
    truth = list(true_ids)
    rankings = list(retrieved_ids)
    if not truth:
        return 0.0
    return sum(expected in list(ranking)[:k] for expected, ranking in zip(truth, rankings, strict=True)) / len(truth)


def rank_statistics(ranks: Sequence[int | None]) -> dict[str, float | int | None]:
    valid = sorted(rank for rank in ranks if rank is not None)
    if not valid:
        return {"median_rank": None, "p95_rank": None, "count": 0}
    return {
        "median_rank": float(median(valid)),
        "p95_rank": float(np.percentile(valid, 95)),
        "count": len(valid),
    }


@dataclass(frozen=True, slots=True)
class OpenSetMetrics:
    tpr: float
    far: float
    accepted_precision: float
    correct_rejection_rate: float


def open_set_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> OpenSetMetrics:
    if len(labels) != len(predictions):
        raise ValueError("labels e predictions devem ter o mesmo tamanho")
    present = [i for i, label in enumerate(labels) if label]
    absent = [i for i, label in enumerate(labels) if not label]
    accepted = [i for i, prediction in enumerate(predictions) if prediction]
    true_accepted = sum(labels[i] and predictions[i] for i in range(len(labels)))
    return OpenSetMetrics(
        tpr=sum(predictions[i] for i in present) / len(present) if present else 0.0,
        far=sum(predictions[i] for i in absent) / len(absent) if absent else 0.0,
        accepted_precision=true_accepted / len(accepted) if accepted else 0.0,
        correct_rejection_rate=sum(not predictions[i] for i in absent) / len(absent) if absent else 0.0,
    )
