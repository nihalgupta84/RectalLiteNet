"""Dice and IoU metrics for the two foreground classes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CLASS_NAMES = {1: "normal_rectum", 2: "tumor"}


def overlap_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    class_id: int,
) -> tuple[int, int, int]:
    """Return true-positive, false-positive, and false-negative pixel counts."""
    predicted = prediction == class_id
    expected = target == class_id
    return (
        int(np.logical_and(predicted, expected).sum()),
        int(np.logical_and(predicted, np.logical_not(expected)).sum()),
        int(np.logical_and(np.logical_not(predicted), expected).sum()),
    )


def summarize_counts(counts: dict[int, list[int]]) -> dict[str, object]:
    """Convert accumulated overlap counts into readable metrics."""
    classes: dict[str, dict[str, float]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        true_positive, false_positive, false_negative = counts[class_id]
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        classes[class_name] = {
            "dice": (
                2 * true_positive / dice_denominator
                if dice_denominator > 0
                else 1.0
            ),
            "iou": true_positive / iou_denominator if iou_denominator > 0 else 1.0,
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
        }
    return {
        "classes": classes,
        "macro_dice": float(np.mean([row["dice"] for row in classes.values()])),
        "macro_iou": float(np.mean([row["iou"] for row in classes.values()])),
    }


@dataclass
class MetricAccumulator:
    """Accumulate pixel counts across images before computing metrics."""

    counts: dict[int, list[int]] = field(
        default_factory=lambda: {class_id: [0, 0, 0] for class_id in CLASS_NAMES}
    )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
            )
        for class_id in CLASS_NAMES:
            values = overlap_counts(prediction, target, class_id)
            for index, value in enumerate(values):
                self.counts[class_id][index] += value

    def compute(self) -> dict[str, object]:
        return summarize_counts(self.counts)
