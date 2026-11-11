"""Kiểm tra và làm sạch tracks.parquet theo schema chung, dùng cho scripts/make_tracks.py và mọi bản tracks khác.

Đầu vào: bảng tracks (video_id, frame_idx, track_id, x1, y1, x2, y2, det_conf; frame_idx 0-based, hộp pixel xyxy)
và metadata. Đầu ra: TrackCheck liệt kê vi phạm theo mã và video không có track.
Ví dụ: check = validate_tracks(pd.read_parquet("tracks.parquet"), load_metadata()); print(check.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .common import TRACK_COLUMNS

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]
FLOAT_COLUMNS = [*BOX_COLUMNS, "det_conf"]
INT_COLUMNS = ["frame_idx", "track_id"]
KEY_COLUMNS = ["video_id", "frame_idx", "track_id"]
MAX_EXAMPLES = 5


@dataclass
class TrackCheck:
    """Kết quả validate_tracks. issues: list (mã, mô tả); empty_videos: video trong metadata không có dòng nào."""

    issues: list[tuple[str, str]] = field(default_factory=list)
    empty_videos: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def codes(self) -> set[str]:
        return {code for code, _ in self.issues}

    def summary(self) -> str:
        lines = [f"[{code}] {text}" for code, text in self.issues] or ["không có vi phạm"]
        lines.append(f"video không có track: {len(self.empty_videos)}"
                     + (f" ({', '.join(self.empty_videos)})" if self.empty_videos else ""))
        return "\n".join(lines)


def empty_tracks() -> pd.DataFrame:
    """Bảng tracks rỗng đúng kiểu, để shard của video không có người vẫn giữ schema."""
    return cast_tracks(pd.DataFrame(columns=TRACK_COLUMNS))


def cast_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu theo schema: video_id chuỗi, frame_idx/track_id int64, hộp và det_conf float64."""
    out = tracks[TRACK_COLUMNS].copy()
    out["video_id"] = out["video_id"].astype(str) if len(out) else out["video_id"].astype(object)
    for column in INT_COLUMNS:
        out[column] = out[column].astype("int64")
    for column in FLOAT_COLUMNS:
        out[column] = out[column].astype("float64")
    return out.reset_index(drop=True)


def clip_to_frame(tracks: pd.DataFrame, width: int, height: int) -> tuple[pd.DataFrame, int]:
    """Clip hộp vào khung [0, width] × [0, height] (pixel); bỏ hộp không còn diện tích. Trả về (bảng, số dòng bỏ)."""
    out = tracks.copy()
    out[["x1", "x2"]] = out[["x1", "x2"]].clip(0, width)
    out[["y1", "y2"]] = out[["y1", "y2"]].clip(0, height)
    keep = (out["x2"] > out["x1"]) & (out["y2"] > out["y1"])
    return out[keep].reset_index(drop=True), int((~keep).sum())


def _examples(values) -> str:
    values = list(values)
    more = f" … (+{len(values) - MAX_EXAMPLES})" if len(values) > MAX_EXAMPLES else ""
    return ", ".join(str(v) for v in values[:MAX_EXAMPLES]) + more


def _dtype_issues(tracks: pd.DataFrame) -> list[tuple[str, str]]:
    issues = []
    if not (ptypes.is_object_dtype(tracks["video_id"]) or ptypes.is_string_dtype(tracks["video_id"])) or \
            not tracks["video_id"].map(lambda v: isinstance(v, str)).all():
        issues.append(("dtype", f"video_id phải là chuỗi, đang là {tracks['video_id'].dtype}"))
    for column in INT_COLUMNS:
        if not ptypes.is_integer_dtype(tracks[column]) or tracks[column].isna().any():
            issues.append(("dtype", f"{column} phải là số nguyên không trống, đang là {tracks[column].dtype}"))
    for column in FLOAT_COLUMNS:
        if not ptypes.is_float_dtype(tracks[column]):
            issues.append(("dtype", f"{column} phải là float, đang là {tracks[column].dtype}"))
        elif not np.isfinite(tracks[column].to_numpy()).all():
            issues.append(("dtype", f"{column} có NaN hoặc vô hạn"))
    return issues


def _frame_and_box_issues(tracks: pd.DataFrame, meta: pd.DataFrame) -> list[tuple[str, str]]:
    """frame_idx ∈ [0, n_frames), hộp trong khung và có diện tích, det_conf ∈ [0, 1]. Chỉ xét dòng có video hợp lệ."""
    info = meta.set_index("video_id")[["n_frames", "width", "height"]].astype(float)
    rows = tracks.assign(video_id=tracks["video_id"].astype(str)).join(info, on="video_id", how="inner")
    issues = []
    bad_frame = (rows["frame_idx"] < 0) | (rows["frame_idx"] >= rows["n_frames"])
    if bad_frame.any():
        issues.append(("frame_range", f"{int(bad_frame.sum())} dòng có frame_idx ngoài [0, n_frames): "
                       + _examples(rows.loc[bad_frame, ["video_id", "frame_idx"]].itertuples(index=False, name=None))))
    outside = (rows["x1"] < 0) | (rows["y1"] < 0) | (rows["x2"] > rows["width"]) | (rows["y2"] > rows["height"])
    if outside.any():
        issues.append(("box_outside", f"{int(outside.sum())} hộp vượt khung: "
                       + _examples(rows.loc[outside, KEY_COLUMNS].itertuples(index=False, name=None))))
    flat = (rows["x2"] <= rows["x1"]) | (rows["y2"] <= rows["y1"])
    if flat.any():
        issues.append(("box_area", f"{int(flat.sum())} hộp có x2 <= x1 hoặc y2 <= y1: "
                       + _examples(rows.loc[flat, KEY_COLUMNS].itertuples(index=False, name=None))))
    bad_conf = (rows["det_conf"] < 0) | (rows["det_conf"] > 1)
    if bad_conf.any():
        issues.append(("det_conf", f"{int(bad_conf.sum())} dòng có det_conf ngoài [0, 1]"))
    return issues


def validate_tracks(tracks: pd.DataFrame, meta: pd.DataFrame) -> TrackCheck:
    """Kiểm tra tracks so với metadata (các video được kỳ vọng). Không sửa dữ liệu, chỉ báo.

    Mã vi phạm: columns, dtype, unknown_video, frame_range, duplicate, box_outside, box_area, det_conf.
    Video trong meta mà không có dòng nào không phải vi phạm (video không có người), được liệt kê ở empty_videos.
    """
    check = TrackCheck()
    missing = [column for column in TRACK_COLUMNS if column not in tracks.columns]
    if missing:
        check.issues.append(("columns", f"thiếu cột {missing}"))
        return check
    check.issues += _dtype_issues(tracks)
    unknown = sorted(set(tracks["video_id"].astype(str)) - set(meta["video_id"]))
    if unknown:
        check.issues.append(("unknown_video", f"{len(unknown)} video_id không có trong metadata: {_examples(unknown)}"))
    duplicated = tracks.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicated.any():
        check.issues.append(("duplicate", f"{int(duplicated.sum())} dòng trùng (video_id, frame_idx, track_id): "
                             + _examples(tracks.loc[duplicated, KEY_COLUMNS].drop_duplicates()
                                         .itertuples(index=False, name=None))))
    if all(ptypes.is_numeric_dtype(tracks[column]) for column in INT_COLUMNS + FLOAT_COLUMNS):
        check.issues += _frame_and_box_issues(tracks, meta)
    check.empty_videos = sorted(set(meta["video_id"]) - set(tracks["video_id"].astype(str)))
    return check
