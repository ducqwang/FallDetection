"""Tín hiệu hình học theo từng track cho M1 và EDA.

Đầu vào: các dòng của 1 track (frame_idx, x1, y1, x2, y2) và fps. Đầu ra: DataFrame theo frame.
Mọi tín hiệu chỉ dùng quá khứ, nên điểm M1 tính được khi video đang chạy.
Vận tốc tính bằng chiều cao người/giây nên không phụ thuộc độ phân giải; y hướng xuống nên vy > 0 là đi xuống.
Ví dụ: signals = track_signals(track_rows, fps=25, smooth_s=0.2, reference_s=1.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]
MIN_SIZE_PX = 1.0  # tránh chia cho 0 khi bbox suy biến


def seconds_to_frames(seconds: float, fps: float) -> int:
    """Đổi giây sang số frame, ít nhất 1."""
    return max(1, round(seconds * fps))


def moving_average(values: pd.Series, window: int) -> pd.Series:
    """Trung bình trượt về quá khứ, `window` tính bằng frame (1 mẫu = 1 frame)."""
    return values.rolling(window, min_periods=1).mean()


def contiguous_track(track: pd.DataFrame) -> pd.DataFrame:
    """Sắp theo frame và nội suy tuyến tính các frame track bị mất giữa chừng.

    Trả về bảng index frame_idx (0-based) liên tục từ frame đầu đến frame cuối của track.
    """
    if track["frame_idx"].duplicated().any():
        raise ValueError(f"Track có frame trùng: {track['frame_idx'][track['frame_idx'].duplicated()].tolist()[:5]}")
    boxes = track.set_index("frame_idx")[BOX_COLUMNS].astype(float).sort_index()
    frames = pd.RangeIndex(boxes.index.min(), boxes.index.max() + 1, name="frame_idx")
    return boxes.reindex(frames).interpolate(method="linear")


def track_signals(track: pd.DataFrame, fps: float, smooth_s: float, reference_s: float) -> pd.DataFrame:
    """Tính tín hiệu cho 1 track.

    fps: frame/giây; smooth_s, reference_s: giây. Trả về bảng index frame_idx (0-based) với các cột:
    aspect (w/h); vy (chiều cao/giây); ay (chiều cao/giây²); speed (tốc độ tâm, chiều cao/giây).
    Chiều cao chuẩn là chiều cao lớn nhất trong reference_s giây gần nhất: bbox thấp đi khi ngã,
    chia cho chiều cao hiện tại sẽ làm vận tốc bị thổi phồng lúc người đã nằm.
    """
    boxes = contiguous_track(track)
    smooth = seconds_to_frames(smooth_s, fps)
    width = (boxes["x2"] - boxes["x1"]).clip(lower=MIN_SIZE_PX)
    height = (boxes["y2"] - boxes["y1"]).clip(lower=MIN_SIZE_PX)
    center_x = moving_average((boxes["x1"] + boxes["x2"]) / 2, smooth)
    center_y = moving_average((boxes["y1"] + boxes["y2"]) / 2, smooth)
    reference = moving_average(height, smooth).rolling(seconds_to_frames(reference_s, fps), min_periods=1).max()
    vy = center_y.diff().fillna(0.0) * fps / reference
    vx = center_x.diff().fillna(0.0) * fps / reference
    return pd.DataFrame({
        "aspect": moving_average(width / height, smooth),
        "vy": vy,
        "ay": moving_average(vy.diff().fillna(0.0) * fps, smooth),
        "speed": np.hypot(vx, vy),
    })
