"""Việc 6: bộ đánh giá mức sự kiện, định nghĩa đầy đủ ở docs/EVAL_PROTOCOL.md.

Đầu vào: predictions.csv và các dòng metadata của tập cần đánh giá.
Đầu ra: dict gồm độ đo sự kiện, độ đo frame, ADL negative khó và kết quả từng video.
Thời gian đổi bằng fps của từng video; cảnh báo là frame đầu của đoạn.
Ví dụ: evaluate(predictions, meta_rows, thr=0.7, tolerance_s=1.0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.common import PREDICTION_COLUMNS
from .frame_metrics import frame_level_metrics, frame_truth

MAIN_FAR_KEY = "far_nonfall_per_hour"  # FAR báo cáo chính; far_per_hour chỉ là số phụ
DEFAULT_THR = 0.5
DEFAULT_MERGE_GAP_S = 1.0
DEFAULT_TOLERANCE_S = 1.0
SECONDS_PER_HOUR = 3600.0
TIME_EPS = 1e-9  # chống sai số làm tròn khi cảnh báo rơi đúng mép cửa sổ


def check_predictions(predictions: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Raise ValueError nếu predictions sai schema, video_id lạ, điểm hoặc frame_idx (0-based) ngoài phạm vi.

    meta: bảng có video_id và n_frames của mọi video được phép xuất hiện.
    """
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions thiếu cột {missing}")
    scores = predictions["fall_score"]
    if scores.isna().any() or (scores < 0).any() or (scores > 1).any():
        bad = predictions[scores.isna() | (scores < 0) | (scores > 1)].iloc[0]
        raise ValueError(f"fall_score phải nằm trong [0, 1] và không được NaN: video {bad['video_id']}, "
                         f"frame {bad['frame_idx']}, điểm {bad['fall_score']}")
    n_frames = meta.set_index("video_id")["n_frames"]
    unknown = sorted(set(predictions["video_id"]) - set(n_frames.index))
    if unknown:
        raise ValueError(f"predictions có video_id không có trong metadata: {unknown[:5]}")
    limits = predictions["video_id"].map(n_frames)
    outside = (predictions["frame_idx"] < 0) | (predictions["frame_idx"] >= limits)
    if outside.any():
        bad = predictions[outside].iloc[0]
        raise ValueError(f"frame_idx ngoài phạm vi: video {bad['video_id']}, frame {bad['frame_idx']}, "
                         f"n_frames {n_frames[bad['video_id']]}")


def max_score_per_frame(video_predictions: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Điểm lớn nhất trên mọi track tại mỗi frame; frame không có track được điểm 0.

    Trả về ndarray độ dài n_frames, chỉ số là frame_idx 0-based.
    """
    scores = np.zeros(n_frames, dtype=float)
    np.maximum.at(scores, video_predictions["frame_idx"].to_numpy(dtype=int),
                  video_predictions["fall_score"].to_numpy(dtype=float))
    return scores


def find_alarms(scores: np.ndarray, fps: float, thr: float,
                merge_gap_s: float = DEFAULT_MERGE_GAP_S) -> list[tuple[int, int]]:
    """Từ điểm đến cảnh báo. Trả về list (frame đầu, frame cuối), 0-based.

    Frame score ≥ thr liên tiếp gộp thành một đoạn. Hai đoạn cách nhau dưới merge_gap_s giây
    (từ frame cuối đoạn trước đến frame đầu đoạn sau) thì gộp làm một.
    """
    above = np.flatnonzero(scores >= thr)
    if len(above) == 0:
        return []
    breaks = np.flatnonzero(np.diff(above) > 1)
    starts = above[np.r_[0, breaks + 1]]
    ends = above[np.r_[breaks, len(above) - 1]]
    merged = [(int(starts[0]), int(ends[0]))]
    for start, end in zip(starts[1:], ends[1:]):
        if (start - merged[-1][1]) / fps < merge_gap_s:
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def match_alarms(alarm_frames: list[int], events: list[tuple[int, int]], fps: float,
                 tolerance_s: float = DEFAULT_TOLERANCE_S) -> dict:
    """So khớp cảnh báo với sự kiện té ngã của 1 video.

    alarm_frames: frame cảnh báo (0-based). events: list (fall_start, fall_end), 0-based.
    Cảnh báo sớm nhất trong [fall_start - T, fall_end + T] là TP; cảnh báo khác cũng trong cửa sổ
    bị bỏ qua (ignored), không tính FP; cảnh báo ngoài mọi cửa sổ là FP; sự kiện không có TP là FN.
    latencies_s = cảnh báo TP - fall_start (giây), giữ giá trị âm.
    """
    matched = [False] * len(events)
    tp = fp = ignored = 0
    latencies: list[float] = []
    for frame in sorted(alarm_frames):
        time_s = frame / fps
        hits = [index for index, (start, end) in enumerate(events)
                if start / fps - tolerance_s - TIME_EPS <= time_s <= end / fps + tolerance_s + TIME_EPS]
        if not hits:
            fp += 1
            continue
        free = [index for index in hits if not matched[index]]
        if free:
            matched[free[0]] = True
            tp += 1
            latencies.append(time_s - events[free[0]][0] / fps)
        else:
            ignored += 1
    return {"tp": tp, "fp": fp, "fn": matched.count(False), "ignored": ignored, "latencies_s": latencies}


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Tỷ số, mẫu bằng 0 thì trả về 0.0."""
    return numerator / denominator if denominator else 0.0


def _event_window_seconds(row: dict, tolerance_s: float) -> float:
    """Độ dài cửa sổ dung sai nằm trong video (giây); dùng cho FAR trên đoạn không-ngã."""
    if not row["events"]:
        return 0.0
    start, end = row["events"][0]
    begin = max(0.0, start / row["fps"] - tolerance_s)
    return min(row["duration_s"], end / row["fps"] + tolerance_s) - begin


def summarize_events(rows: list[dict], tolerance_s: float) -> dict:
    """Gộp kết quả các video (micro): Precision, Recall, F1, FAR, latency.

    tolerance_s tính bằng giây; rows là kết quả từng video (frame 0-based, duration_s bằng giây).

    Không có cảnh báo hoặc không có sự kiện thì độ đo tương ứng bằng 0.0.
    FAR chính far_nonfall_per_hour chia giờ video NGOÀI cửa sổ dung sai; far_per_hour chia tổng giờ, là số phụ.
    """
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    latencies = [value for row in rows for value in row["latencies_s"]]
    seconds = sum(row["duration_s"] for row in rows)
    nonfall_seconds = seconds - sum(_event_window_seconds(row, tolerance_s) for row in rows)
    precision, recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, tp + fn)
    return {
        "n_videos": len(rows), "n_events": tp + fn, "tp": tp, "fp": fp, "fn": fn,
        "ignored_alarms": sum(row["ignored"] for row in rows),
        "precision": precision, "recall": recall, "f1": _safe_ratio(2 * precision * recall, precision + recall),
        "total_hours": seconds / SECONDS_PER_HOUR,
        "far_per_hour": _safe_ratio(fp, seconds / SECONDS_PER_HOUR),
        "far_nonfall_per_hour": _safe_ratio(fp, nonfall_seconds / SECONDS_PER_HOUR),
        "latency_mean_s": float(np.mean(latencies)) if latencies else None,
        "latency_median_s": float(np.median(latencies)) if latencies else None,
    }


def summarize_negatives(rows: list[dict]) -> dict:
    """Báo cáo trên video không có sự kiện (ví dụ ADL negative khó): chỉ có FP và FAR.

    total_hours tính bằng giờ, far_per_hour là số FP mỗi giờ.
    """
    seconds = sum(row["duration_s"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    return {"n_videos": len(rows), "n_videos_with_fp": sum(1 for row in rows if row["fp"] > 0), "fp": fp,
            "total_hours": seconds / SECONDS_PER_HOUR, "far_per_hour": _safe_ratio(fp, seconds / SECONDS_PER_HOUR)}


def _video_result(row: pd.Series, scores: np.ndarray, thr: float, merge_gap_s: float, tolerance_s: float) -> dict:
    """Kết quả 1 video: cảnh báo, TP/FP/FN, thời lượng."""
    fps = float(row["fps"])
    events = [] if row["label"] == "adl" else [(int(row["fall_start_frame"]), int(row["fall_end_frame"]))]
    alarms = [start for start, _ in find_alarms(scores, fps, thr, merge_gap_s)]
    return {"video_id": row["video_id"], "fps": fps, "duration_s": float(row["duration_s"]), "events": events,
            "adl_hard_neg": int(row["adl_hard_neg"]), "alarm_frames": alarms,
            **match_alarms(alarms, events, fps, tolerance_s)}


def _frame_record(result: dict, scores: np.ndarray, frame_labels: pd.DataFrame | None) -> dict:
    """Dữ liệu mức frame của 1 video; ADL chỉ giữ frame có nhãn khi có frame_labels."""
    interval = result["events"][0] if result["events"] else None
    mask = np.ones(len(scores), dtype=bool)
    if interval is None and frame_labels is not None:
        labeled = frame_labels.loc[frame_labels["video_id"] == result["video_id"], "frame_idx"].to_numpy(dtype=int)
        mask = np.zeros(len(scores), dtype=bool)
        mask[labeled] = True
    return {"scores": scores, "truth": frame_truth(len(scores), interval), "mask": mask}


def evaluate(predictions: pd.DataFrame, meta_rows: pd.DataFrame, thr: float = DEFAULT_THR,
             tolerance_s: float = DEFAULT_TOLERANCE_S, merge_gap_s: float = DEFAULT_MERGE_GAP_S,
             frame_labels: pd.DataFrame | None = None, reference_meta: pd.DataFrame | None = None) -> dict:
    """Đánh giá các video trong meta_rows; video không có dòng dự đoán tính là không cảnh báo.

    thr ∈ [0, 1]; tolerance_s, merge_gap_s tính bằng giây; frame_idx 0-based.
    predictions được kiểm tra tại đây, đây là chỗ kiểm tra duy nhất. reference_meta (mặc định meta_rows) là
    bảng dùng để kiểm tra: truyền toàn bộ metadata nếu predictions có cả video ngoài tập đánh giá;
    các video đó hợp lệ và bị bỏ qua khi tính độ đo.
    Video has_event_label = 0 không được đánh giá. Trả về dict: params, event, frame,
    hard_negative, per_video. frame_labels (urfd_frame_labels) bật lọc frame có nhãn ở ADL.
    """
    if (meta_rows["has_event_label"] != 1).any():
        raise ValueError("Không đánh giá video has_event_label = 0")
    check_predictions(predictions, meta_rows if reference_meta is None else reference_meta)
    in_scope = predictions[predictions["video_id"].isin(meta_rows["video_id"])]
    groups = dict(tuple(in_scope.groupby("video_id"))) if len(in_scope) else {}
    empty = predictions.iloc[0:0]
    results, records = [], []
    for _, row in meta_rows.iterrows():
        scores = max_score_per_frame(groups.get(row["video_id"], empty), int(row["n_frames"]))
        result = _video_result(row, scores, thr, merge_gap_s, tolerance_s)
        results.append(result)
        records.append(_frame_record(result, scores, frame_labels))
    per_video = [{key: item[key] for key in ("video_id", "alarm_frames", "tp", "fp", "fn", "ignored", "latencies_s")}
                 for item in results]
    return {"params": {"thr": thr, "tolerance_s": tolerance_s, "merge_gap_s": merge_gap_s},
            "event": summarize_events(results, tolerance_s), "frame": frame_level_metrics(records, thr),
            "hard_negative": summarize_negatives([item for item in results if item["adl_hard_neg"] == 1]),
            "per_video": per_video}
