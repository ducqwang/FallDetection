"""Độ đo phụ mức frame: Accuracy, ROC-AUC, PR-AUC.

Đầu vào: điểm theo frame từng video và khoảng ngã. Đầu ra: dict độ đo (None nếu thiếu một lớp).
Frame té ngã là frame trong [fall_start, fall_end]; mọi frame ADL là bình thường.
Ví dụ: frame_level_metrics(records, thr=0.5).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def frame_truth(n_frames: int, fall_interval: tuple[int, int] | None) -> np.ndarray:
    """Mảng nhãn 0/1 độ dài n_frames; fall_interval là (đầu, cuối) theo frame 0-based, tính cả hai đầu, None với ADL."""
    truth = np.zeros(n_frames, dtype=int)
    if fall_interval is not None:
        truth[fall_interval[0]:fall_interval[1] + 1] = 1
    return truth


def frame_level_metrics(records: list[dict], thr: float) -> dict:
    """Tính độ đo trên mọi frame được giữ lại.

    records: list dict gồm scores, truth (ndarray theo frame, 0-based) và mask (True = dùng frame này).
    Mask dùng để bỏ frame NaN của ADL khi đánh giá theo nhãn frame.
    """
    scores = np.concatenate([item["scores"][item["mask"]] for item in records])
    truth = np.concatenate([item["truth"][item["mask"]] for item in records])
    both_classes = len(np.unique(truth)) == 2
    return {
        "n_frames": int(len(truth)), "n_fall_frames": int(truth.sum()),
        "accuracy": float(accuracy_score(truth, scores >= thr)),
        "roc_auc": float(roc_auc_score(truth, scores)) if both_classes else None,
        "pr_auc": float(average_precision_score(truth, scores)) if both_classes else None,
    }
