"""Việc 3: gán nhãn cửa sổ và chọn track chính, dùng chung cho M2, M3.

Đầu vào: 1 dòng metadata, khoảng frame, bảng tracks. Đầu ra: nhãn 0/1 hoặc track_id.
Mọi frame ở đây là frame_idx 0-based.
Ví dụ: window_label(row, 100, 131); select_main_track(tracks_of_video, row, gt_of_video).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .common import drop_duplicate_frames


def window_label(video_row: Mapping, start_frame: int, end_frame: int, min_overlap: float = 0.5) -> int:
    """Nhãn của cửa sổ [start_frame, end_frame] (frame 0-based, tính cả hai đầu).

    Trả về 1 nếu số frame chồng lên sự kiện [fall_start, fall_end] (tính cả hai đầu)
    ≥ min_overlap × min(độ dài cửa sổ, độ dài sự kiện).
    Dùng min vì cú ngã có thể ngắn hơn cửa sổ nhiều: chia theo độ dài cửa sổ thì cửa sổ dài hơn
    2 lần cú ngã không bao giờ dương, dù đã bao trọn cú ngã (Le2i có cú ngã chỉ 7 frame).
    Video ADL luôn trả về 0, kể cả ADL có nhãn frame 0/1. Video chưa xác định lớp thì raise.
    """
    if end_frame < start_frame:
        raise ValueError(f"Cửa sổ không hợp lệ: [{start_frame}, {end_frame}]")
    if video_row["label"] == "adl":
        return 0
    if video_row["label"] != "fall":
        raise ValueError(f"{video_row['video_id']}: chưa xác định lớp, không dùng để gán nhãn")
    fall_start, fall_end = int(video_row["fall_start_frame"]), int(video_row["fall_end_frame"])
    overlap = min(end_frame, fall_end) - max(start_frame, fall_start) + 1
    window_length = end_frame - start_frame + 1
    event_length = fall_end - fall_start + 1
    return int(overlap > 0 and overlap >= min_overlap * min(window_length, event_length))


def bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU từng cặp của 2 mảng N×4 (x1, y1, x2, y2), pixel."""
    x1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)


def _longest_track(tracks_df: pd.DataFrame) -> int:
    """Track có nhiều frame nhất; hòa thì lấy track_id nhỏ nhất."""
    counts = tracks_df.groupby("track_id").size()
    return int(counts[counts == counts.max()].index.min())


def _best_iou_track(tracks_df: pd.DataFrame, gt_bbox_df: pd.DataFrame, width: int, height: int) -> int:
    """Track có IoU trung bình cao nhất so với GT.

    Mẫu số là số frame có GT của video (frame track thiếu tính IoU = 0), nên track rất ngắn
    không thắng được track bám người suốt video. Hòa thì lấy track_id nhỏ nhất.
    """
    scores: dict[int, float] = {}
    gt = drop_duplicate_frames(gt_bbox_df)[0].set_index("frame_idx")[["x1", "y1", "x2", "y2"]]
    for track_id, group in tracks_df.groupby("track_id"):
        shared = group[group["frame_idx"].isin(gt.index)]
        boxes = shared[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)  # clip như GT trước khi tính IoU
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        ious = bbox_iou(boxes, gt.loc[shared["frame_idx"]].to_numpy(dtype=float))
        scores[int(track_id)] = float(ious.sum()) / len(gt)
    best = max(scores.values())
    return min(track_id for track_id, score in scores.items() if score == best)


def select_main_track(tracks_df: pd.DataFrame, video_row: Mapping,
                      gt_bbox_df: pd.DataFrame | None = None) -> int | None:
    """Chọn track_id chính của 1 video; các track còn lại coi là normal.

    Le2i: IoU trung bình cao nhất so với GT (gt_bbox_df của đúng video này).
    URFD: track tồn tại lâu nhất. Không có track thì trả về None.
    Le2i không có dòng GT nào (trường hợp hiếm) thì dùng track dài nhất.
    """
    if tracks_df.empty:
        return None
    if video_row["dataset"] == "le2i":
        if gt_bbox_df is None:
            raise ValueError(f"{video_row['video_id']}: Le2i cần gt_bbox_df")
        if not gt_bbox_df.empty:
            return _best_iou_track(tracks_df, gt_bbox_df, int(video_row["width"]), int(video_row["height"]))
    return _longest_track(tracks_df)
