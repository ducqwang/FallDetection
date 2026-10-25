"""Độ đo đánh giá THỐNG NHẤT cho cả 3 mô hình.

Trọng tâm: mức SỰ KIỆN với cửa sổ dung sai ±T giây (định nghĩa chốt trong kế hoạch).
Một cảnh báo là TP nếu rơi trong [fall_start - T, fall_end + T]; ngoài ra là FP;
sự kiện ngã không có cảnh báo nào trùng = FN.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EventMetrics:
    precision: float
    recall: float
    f1: float
    false_alarm_per_hour: float
    mean_latency_sec: float
    tp: int
    fp: int
    fn: int


def evaluate_events(pred_alarms, gt_falls, video_seconds: float,
                    fps: float, tolerance_sec: float = 1.0) -> EventMetrics:
    """So khớp cảnh báo dự đoán với sự kiện ngã thật ở mức sự kiện.

    pred_alarms: list[frame_index] các frame phát cảnh báo (đã gộp cụm liền kề).
    gt_falls:    list[(start_frame, end_frame)] các sự kiện ngã thật.
    """
    tol = tolerance_sec * fps
    matched_gt = set()
    tp = 0
    latencies = []
    for a in sorted(pred_alarms):
        hit = None
        for gi, (s, e) in enumerate(gt_falls):
            if gi in matched_gt:
                continue
            if s - tol <= a <= e + tol:
                hit = gi
                latencies.append(max(0.0, (a - s) / fps))
                break
        if hit is not None:
            matched_gt.add(hit)
            tp += 1
    fp = len(pred_alarms) - tp
    fn = len(gt_falls) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (video_seconds / 3600) if video_seconds else 0.0
    lat = sum(latencies) / len(latencies) if latencies else 0.0
    return EventMetrics(precision, recall, f1, far, lat, tp, fp, fn)


def merge_alarms(flags, min_gap: int = 5):
    """Gộp các frame cảnh báo liên tiếp thành 1 sự kiện (lấy frame đầu mỗi cụm)."""
    frames = [i for i, v in enumerate(flags) if v]
    if not frames:
        return []
    events, prev = [frames[0]], frames[0]
    for f in frames[1:]:
        if f - prev > min_gap:
            events.append(f)
        prev = f
    return events


def aggregate_over_dataset(per_video: list[EventMetrics], video_seconds_total: float) -> EventMetrics:
    """Gộp EventMetrics trên nhiều video (micro: cộng dồn tp/fp/fn)."""
    tp = sum(m.tp for m in per_video)
    fp = sum(m.fp for m in per_video)
    fn = sum(m.fn for m in per_video)
    lat = [m.mean_latency_sec for m in per_video if m.tp > 0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (video_seconds_total / 3600) if video_seconds_total else 0.0
    mean_lat = sum(lat) / len(lat) if lat else 0.0
    return EventMetrics(precision, recall, f1, far, mean_lat, tp, fp, fn)


def results_table(rows: list[dict], out_csv: str | None = None):
    """Tạo bảng so sánh (model x protocol) từ các dòng kết quả; xuất CSV cho báo cáo.

    rows: [{'model':..., 'protocol':..., 'precision':..., 'recall':..., 'f1':...,
            'false_alarm_per_hour':..., 'mean_latency_sec':..., 'fps':...}, ...]
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    if out_csv:
        from pathlib import Path
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df
