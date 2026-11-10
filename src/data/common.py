"""Quy ước chung của nhóm: tên cột, video_id và lỗi dữ liệu (mục 3 của prompt).

Đầu vào: không có. Đầu ra: hằng số và hàm nhỏ cho các module trong src/data.
Ví dụ: urfd_video_id("fall-05", "cam0") trả về "urfd_fall-05_cam0".
"""
from __future__ import annotations

import re

import pandas as pd

METADATA_COLUMNS = [
    "video_id", "dataset", "scene", "scene_part", "camera", "label",
    "fall_start_frame", "fall_end_frame", "fps", "n_frames",
    "width", "height", "duration_s", "has_event_label",
    "label_first_frame", "label_last_frame", "label_coverage",
    "adl_hard_neg", "path", "label_source",
]
GT_BBOX_COLUMNS = ["video_id", "frame_idx", "x1", "y1", "x2", "y2", "aux_code"]
FRAME_LABEL_COLUMNS = ["video_id", "frame_idx", "label"]
TRACK_COLUMNS = ["video_id", "frame_idx", "track_id", "x1", "y1", "x2", "y2", "det_conf"]
PREDICTION_COLUMNS = ["video_id", "frame_idx", "track_id", "fall_score"]

# Cột số nguyên có thể trống: dùng Int64 để CSV ghi 82 chứ không phải 82.0.
NULLABLE_INT_COLUMNS = ["fall_start_frame", "fall_end_frame", "label_first_frame", "label_last_frame"]
# scene_part phải đọc là chuỗi, nếu không "01" sẽ thành số 1.
METADATA_DTYPES = {"video_id": str, "dataset": str, "scene": str, "scene_part": str, "camera": str,
                   "label": str, "path": str, "label_source": str, **{c: "Int64" for c in NULLABLE_INT_COLUMNS}}

_URFD_ID_RE = re.compile(r"^urfd_((?:fall|adl)-\d+)_cam\d+$")


class DataCheckError(RuntimeError):
    """Dữ liệu thật lệch với mô tả đã kiểm tra: dừng lại và báo TV1."""


def urfd_video_id(seq: str, camera: str) -> str:
    """Tạo video_id URFD. seq dạng 'fall-05', camera dạng 'cam0'."""
    return f"urfd_{seq}_{camera}"


def le2i_video_id(scene: str, part: str, number: int) -> str:
    """Tạo video_id Le2i, ví dụ ('home', '02', 31) -> 'le2i_home02_031'."""
    return f"le2i_{scene}{part}_{number:03d}"


def sequence_id(video_id: str) -> str:
    """Khóa nhóm chống rò rỉ: bỏ camera của URFD, Le2i giữ nguyên video_id.

    Ví dụ urfd_fall-05_cam0 và urfd_fall-05_cam1 cùng trả về urfd_fall-05.
    """
    match = _URFD_ID_RE.match(video_id)
    return f"urfd_{match.group(1)}" if match else video_id


def drop_duplicate_frames(boxes: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Giữ dòng đầu tiên (theo thứ tự file) của mỗi (video_id, frame_idx[, track_id]); trả về (bảng, số dòng bị bỏ).

    Le2i GT thật (`le2i_gt_bbox.csv`) có 350 dòng ghi trùng frame, có khi 2 hộp khác nhau, không rõ hộp nào đúng.
    Mọi chỗ đọc GT gọi hàm này để cùng một quy tắc; file do prepare_data sinh ra giữ nguyên.
    """
    keys = [column for column in ("video_id", "track_id", "frame_idx") if column in boxes]
    kept = boxes.drop_duplicates(subset=keys, keep="first")
    return kept.reset_index(drop=True), len(boxes) - len(kept)
