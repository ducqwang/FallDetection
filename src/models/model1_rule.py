"""Mô hình 1 — Rule-based (hình học). KHÔNG huấn luyện.

Phát cờ té ngã dựa trên: khung bao chuyển từ đứng (cao) sang nằm (rộng),
vận tốc dọc của tâm tăng đột ngột, sau đó có giai đoạn ít chuyển động.
Ngưỡng được tinh chỉnh trên tập val (grid search đơn giản).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RuleThresholds:
    ar_fall: float = 1.2        # w/h vượt ngưỡng -> nghi nằm
    vy_fall: float = 8.0        # vận tốc dọc (px/frame) đột ngột
    still_ratio: float = 0.4    # tỷ lệ frame ít chuyển động sau ngã


def predict_sequence(track_seq, th: RuleThresholds) -> list[bool]:
    """Trả về cờ cảnh báo theo từng frame của một ID người.

    track_seq: list[Track] của cùng một track_id, sắp theo frame.
    """
    flags = [False] * len(track_seq)
    for i in range(2, len(track_seq)):
        cy = [track_seq[j].y + track_seq[j].h / 2 for j in range(i - 2, i + 1)]
        vy = cy[-1] - cy[-2]
        ar = track_seq[i].w / max(track_seq[i].h, 1e-6)
        if ar > th.ar_fall and vy > th.vy_fall:
            flags[i] = True
    return flags


def tune_thresholds(val_samples, grid=None) -> RuleThresholds:
    """Grid search ngưỡng trên tập val để tối đa F1 mức sự kiện.

    val_samples: iterable các dict {tracks_by_id, gt_falls, fps, seconds}, trong đó
    tracks_by_id = list[list[Track]] (mỗi phần tử là chuỗi của một ID).
    """
    from ..evaluate import merge_alarms, evaluate_events

    if grid is None:
        grid = {
            "ar_fall": [0.9, 1.0, 1.1, 1.2, 1.4],
            "vy_fall": [4.0, 6.0, 8.0, 10.0, 14.0],
            "still_ratio": [0.3, 0.4, 0.5],
        }
    best, best_f1 = RuleThresholds(), -1.0
    for ar in grid["ar_fall"]:
        for vy in grid["vy_fall"]:
            for sr in grid["still_ratio"]:
                th = RuleThresholds(ar_fall=ar, vy_fall=vy, still_ratio=sr)
                tp = fp = fn = 0
                for s in val_samples:
                    alarms = []
                    for seq in s["tracks_by_id"]:
                        alarms += merge_alarms(predict_sequence(seq, th))
                    m = evaluate_events(alarms, s["gt_falls"], s["seconds"], s["fps"])
                    tp += m.tp; fp += m.fp; fn += m.fn
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                if f1 > best_f1:
                    best_f1, best = f1, th
    print(f"Ngưỡng tốt nhất trên val: {best} (F1={best_f1:.4f})")
    return best
