"""Việc 3: hàm đọc dữ liệu đã chuẩn hóa, dùng chung cho M1, M2, M3.

Đầu vào: thư mục nhom05-falldet-processed. Đầu ra: DataFrame, frame numpy, Series nhãn.
Thư mục được tìm theo thứ tự: tham số root, biến môi trường FALLDET_DATA_ROOT, /kaggle/working, /kaggle/input.
Ví dụ: from src.data.loader import load_metadata, get_split, read_frames; ids = get_split("urfd", "test").
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .common import METADATA_DTYPES, TRACK_COLUMNS

ENV_ROOT = "FALLDET_DATA_ROOT"
DATASET_NAME = "nhom05-falldet-processed"
KAGGLE_ROOTS = (Path("/kaggle/working"), Path("/kaggle/input"))
BGR_CHANNELS = 3


def data_root(root: str | Path | None = None) -> Path:
    """Trả về thư mục dữ liệu đã chuẩn hóa; không tìm thấy thì raise."""
    if root is not None:
        return Path(root)
    if os.environ.get(ENV_ROOT):
        return Path(os.environ[ENV_ROOT])
    for base in KAGGLE_ROOTS:
        if base.exists():
            hits = sorted(base.rglob(f"{DATASET_NAME}/metadata.csv"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError(f"Không thấy {DATASET_NAME}. Truyền root hoặc đặt biến {ENV_ROOT}")


def load_metadata(root: str | Path | None = None) -> pd.DataFrame:
    """Đọc metadata.csv. scene_part giữ dạng chuỗi ('01'); cột frame trống là <NA>."""
    return pd.read_csv(data_root(root) / "metadata.csv", dtype=METADATA_DTYPES)


def get_split(dataset: str, part: str, root: str | Path | None = None) -> list[str]:
    """Trả về list video_id. dataset: 'urfd'|'le2i'; part: 'train'|'val'|'test'."""
    splits = json.loads((data_root(root) / "splits.json").read_text(encoding="utf-8"))
    return splits[dataset][part]


def _video_row(video_id: str, root: Path) -> pd.Series:
    meta = load_metadata(root)
    rows = meta[meta["video_id"] == video_id]
    if rows.empty:
        raise KeyError(f"video_id không có trong metadata: {video_id}")
    return rows.iloc[0]


def read_frames(video_id: str, start: int | None = None, end: int | None = None,
                root: str | Path | None = None) -> np.ndarray:
    """Đọc frame [start, end) của 1 video, trả về ndarray T×H×W×3 BGR uint8.

    start, end là frame_idx 0-based, end không tính. None nghĩa là đầu/cuối video.
    Giải mã bằng ffmpeg trong tiến trình riêng, trả đúng số frame yêu cầu.
    """
    base = data_root(root)
    row = _video_row(video_id, base)
    first = 0 if start is None else start
    last = int(row["n_frames"]) if end is None else min(end, int(row["n_frames"]))
    if not 0 <= first <= last:
        raise ValueError(f"{video_id}: khoảng frame không hợp lệ [{first}, {last})")
    width, height = int(row["width"]), int(row["height"])
    # setpts đánh lại thời gian để ffmpeg không nhân đôi frame sau khi lọc.
    video_filter = f"select='between(n,{first},{last - 1})',setpts=N/FRAME_RATE/TB"
    cmd = ["ffmpeg", "-v", "error", "-i", str(base / row["path"]), "-vf", video_filter,
           "-frames:v", str(last - first), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    done = subprocess.run(cmd, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"{video_id}: ffmpeg lỗi: {done.stderr[-300:]!r}")
    frames = np.frombuffer(done.stdout, dtype=np.uint8).reshape(-1, height, width, BGR_CHANNELS)
    if len(frames) != last - first:
        raise RuntimeError(f"{video_id}: đọc được {len(frames)} frame, cần {last - first}")
    return frames


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Đọc tracks.parquet (frame_idx 0-based, bbox pixel) và kiểm tra đủ cột theo schema."""
    tracks = pd.read_parquet(path)
    missing = sorted(set(TRACK_COLUMNS) - set(tracks.columns))
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing}")
    return tracks[TRACK_COLUMNS]


def load_frame_labels(video_id: str, root: str | Path | None = None) -> pd.Series:
    """Nhãn frame của 1 video URFD: Series index frame_idx 0..n_frames-1, frame thiếu là NaN.

    Giá trị -1/0/1 như CSV gốc. Không tự điền nhãn cho frame thiếu.
    """
    base = data_root(root)
    labels = pd.read_csv(base / "annotations" / "urfd_frame_labels.csv")
    row = _video_row(video_id, base)
    own = labels[labels["video_id"] == video_id].set_index("frame_idx")["label"].astype(float)
    index = pd.RangeIndex(int(row["n_frames"]), name="frame_idx")
    return own.reindex(index).rename("label")
