"""Đặc trưng hình học theo cửa sổ cho biểu đồ PCA/t-SNE (bản giữa kỳ).

Đầu vào: bbox (Le2i GT hoặc tracks cache) theo video và metadata. Đầu ra: X (N×D), y (N), danh sách tên cột.
Mỗi cửa sổ: thống kê w/h, vận tốc và gia tốc dọc của tâm bbox, mức bất động.
Mỗi video chỉ lấy 1 track chính (labels.select_main_track), trừ khi nguồn là GT vốn có 1 track.
Ví dụ: X, y, names = build_window_features(boxes, meta, window_s=1.0, stride_s=0.25, signal_params=p, still_speed=0.15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.labels import select_main_track, window_label
from ..models.m1_features import seconds_to_frames, track_signals

FEATURE_NAMES = ["aspect_mean", "aspect_std", "aspect_min", "aspect_max", "vy_max", "vy_mean",
                 "ay_max", "speed_mean", "still_ratio"]


def window_vector(signals: pd.DataFrame, still_speed: float) -> np.ndarray:
    """Vector đặc trưng của 1 cửa sổ tín hiệu, thứ tự theo FEATURE_NAMES."""
    aspect = signals["aspect"]
    return np.array([aspect.mean(), aspect.std(ddof=0), aspect.min(), aspect.max(), signals["vy"].max(),
                     signals["vy"].mean(), signals["ay"].max(), signals["speed"].mean(),
                     (signals["speed"] < still_speed).mean()], dtype=float)


def video_windows(track: pd.DataFrame, row: pd.Series, window_s: float, stride_s: float,
                  signal_params: dict, still_speed: float) -> tuple[list[np.ndarray], list[int]]:
    """Cắt 1 track thành các cửa sổ, gán nhãn window_label.

    window_s, stride_s tính bằng giây, đổi sang frame bằng fps của video; frame_idx 0-based.
    """
    fps = float(row["fps"])
    window, stride = seconds_to_frames(window_s, fps), seconds_to_frames(stride_s, fps)
    signals = track_signals(track, fps, signal_params["smooth_s"], signal_params["reference_s"])
    vectors, labels = [], []
    for begin in range(0, len(signals) - window + 1, stride):
        part = signals.iloc[begin:begin + window]
        vectors.append(window_vector(part, still_speed))
        labels.append(window_label(row, int(part.index[0]), int(part.index[-1])))
    return vectors, labels


def build_window_features(boxes: pd.DataFrame, meta: pd.DataFrame, window_s: float, stride_s: float,
                          signal_params: dict, still_speed: float,
                          gt_boxes: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ghép đặc trưng mọi video has_event_label = 1 có bbox.

    boxes cần cột video_id, frame_idx (0-based), x1, y1, x2, y2 (pixel); có track_id thì chọn track chính.
    window_s, stride_s tính bằng giây; still_speed tính bằng chiều cao người/giây.
    gt_boxes dùng để chọn track chính của Le2i khi boxes là tracks cache.
    """
    usable = meta[meta["has_event_label"] == 1].set_index("video_id", drop=False)
    all_vectors: list[np.ndarray] = []
    all_labels: list[int] = []
    for video_id, track in boxes[boxes["video_id"].isin(usable.index)].groupby("video_id", sort=True):
        row = usable.loc[video_id]
        if "track_id" in track:
            gt_of_video = None if gt_boxes is None else gt_boxes[gt_boxes["video_id"] == video_id]
            track = track[track["track_id"] == select_main_track(track, row, gt_of_video)]
        vectors, labels = video_windows(track, row, window_s, stride_s, signal_params, still_speed)
        all_vectors += vectors
        all_labels += labels
    if not all_vectors:
        raise ValueError("Không có cửa sổ nào: kiểm tra nguồn bbox và window_s")
    return np.vstack(all_vectors), np.array(all_labels, dtype=int), FEATURE_NAMES
