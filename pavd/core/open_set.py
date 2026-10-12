from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pavd.config import OpenSetConfig
from pavd.core.metrics import OpenSetMetrics, open_set_metrics


def is_present(scores: Sequence[float], config: OpenSetConfig) -> bool:
    if not scores:
        return False
    s1 = float(scores[0])
    s2 = float(scores[1]) if len(scores) > 1 else 0.0
    return s1 >= config.absolute_threshold or (
        s1 >= config.secondary_threshold
        and s1 - s2 >= config.margin_threshold
    )


@dataclass(slots=True)
class OpenSetEvaluator:
    config: OpenSetConfig

    def predict(self, scores: Sequence[float]) -> bool:
        return is_present(scores, self.config)

    def evaluate(self, labels: Sequence[bool], score_lists: Sequence[Sequence[float]]) -> OpenSetMetrics:
        predictions = [self.predict(scores) for scores in score_lists]
        return open_set_metrics(labels, predictions)
