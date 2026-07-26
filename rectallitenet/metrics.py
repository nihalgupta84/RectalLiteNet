from __future__ import annotations

import numpy as np


CLASS_NAMES = {1: "normal_rectum", 2: "tumor"}


def overlap_counts(
    prediction: np.ndarray, target: np.ndarray, class_id: int
) -> tuple[int, int, int]:
    predicted = prediction == class_id
    expected = target == class_id
    return (
        int(np.logical_and(predicted, expected).sum()),
        int(np.logical_and(predicted, ~expected).sum()),
        int(np.logical_and(~predicted, expected).sum()),
    )


def summarize_counts(
    counts: dict[int, list[int]],
) -> dict[str, object]:
    classes = {}
    for class_id, name in CLASS_NAMES.items():
        true_positive, false_positive, false_negative = counts[class_id]
        dice_denominator = 2 * true_positive + false_positive + false_negative
        iou_denominator = true_positive + false_positive + false_negative
        classes[name] = {
            "dice": 2 * true_positive / dice_denominator,
            "iou": true_positive / iou_denominator,
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, true_positive + false_negative),
        }
    return {
        "classes": classes,
        "macro_dice": float(np.mean([row["dice"] for row in classes.values()])),
        "macro_iou": float(np.mean([row["iou"] for row in classes.values()])),
    }
