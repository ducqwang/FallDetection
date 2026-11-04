"""Dựng các dòng metadata.csv theo schema ở docs/SCHEMA.md.

Đầu vào: dict chuỗi URFD đã đối chiếu (urfd.match_labels) và dict video Le2i đã đọc.
Đầu ra: DataFrame đúng thứ tự cột METADATA_COLUMNS; path tương đối so với out_root.
Ví dụ: table = metadata_frame([urfd_row(s, 30) for s in sequences]).
"""
from __future__ import annotations

import pandas as pd

from .common import METADATA_COLUMNS, NULLABLE_INT_COLUMNS, le2i_video_id, urfd_video_id

URFD_SCENE = "urfd"
VIDEO_DIR = "videos"
COVERAGE_DECIMALS = 4


def _video_path(video_id: str) -> str:
    return f"{VIDEO_DIR}/{video_id}.mp4"


def urfd_row(sequence: dict, fps: int) -> dict:
    """1 dòng metadata URFD. Fall: fall_start/end về 0-based; ADL: để trống, chỉ nhãn mức video."""
    video_id = urfd_video_id(sequence["seq"], sequence["camera"])
    n_frames, is_fall = sequence["n_images"], sequence["label"] == "fall"
    return {
        "video_id": video_id, "dataset": "urfd", "scene": URFD_SCENE, "scene_part": "",
        "camera": sequence["camera"], "label": sequence["label"],
        "fall_start_frame": sequence["first_label0"] - 1 if is_fall else None,
        "fall_end_frame": sequence["first_label1"] - 1 if is_fall else None,
        "fps": fps, "n_frames": n_frames, "width": sequence["width"], "height": sequence["height"],
        "duration_s": n_frames / fps, "has_event_label": 1,
        "label_first_frame": sequence["label_first_frame"],
        "label_last_frame": sequence["label_last_frame"],
        "label_coverage": round(sequence["n_csv_rows"] / n_frames, COVERAGE_DECIMALS),
        "adl_hard_neg": int(not is_fall and sequence["n_label0"] + sequence["n_label1"] > 0),
        "path": _video_path(video_id),
    }


def le2i_row(video: dict) -> dict:
    """1 dòng metadata Le2i. ODD hoặc thiếu annotation: label trống, has_event_label = 0."""
    video_id = le2i_video_id(video["scene"], video["part"], video["number"])
    kind = video["kind"]
    return {
        "video_id": video_id, "dataset": "le2i", "scene": video["scene"], "scene_part": video["part"],
        "camera": "", "label": kind if kind in ("fall", "adl") else "",
        "fall_start_frame": video["start"] - 1 if kind == "fall" else None,
        "fall_end_frame": video["end"] - 1 if kind == "fall" else None,
        "fps": video["fps"], "n_frames": video["n_frames"],
        "width": video["width"], "height": video["height"],
        "duration_s": video["n_frames"] / video["fps"],
        "has_event_label": int(kind in ("fall", "adl")),
        "label_first_frame": None, "label_last_frame": None, "label_coverage": None,
        "adl_hard_neg": 0, "path": _video_path(video_id),
    }


def metadata_frame(rows: list[dict]) -> pd.DataFrame:
    """Gom dòng thành bảng đúng schema; cột frame có thể trống dùng kiểu Int64."""
    table = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    for column in NULLABLE_INT_COLUMNS:
        table[column] = pd.to_numeric(table[column]).astype("Int64")
    if table["video_id"].duplicated().any():
        raise ValueError(f"video_id trùng: {table.loc[table['video_id'].duplicated(), 'video_id'].tolist()[:5]}")
    return table
