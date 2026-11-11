"""Phần lõi của scripts/make_tracks.py: tracking 1 video, làm sạch theo metadata, ghi/đọc shard.

Đầu vào: model có hàm track() kiểu ultralytics (hoặc mock), 1 dòng metadata, thư mục dataset.
Đầu ra: bảng tracks đúng schema (frame_idx 0-based = thứ tự frame đọc được, hộp pixel xyxy) và dict thống kê.
Ví dụ: tracks, stats = process_video(model, row, root, {"conf": 0.25, "tracker": "bytetrack.yaml"})
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import TRACK_COLUMNS
from .tracks import cast_tracks, clip_to_frame, empty_tracks

PERSON_CLASS = 0
MAX_FRAME_DIFF = 1  # lệch số frame lớn hơn mức này so với metadata thì coi là lỗi


def _as_list(values) -> list:
    """Tensor (torch) hoặc mảng (mock) thành list Python."""
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    return np.asarray(values).tolist()


def reset_trackers(model) -> None:
    """Xóa trạng thái ByteTrack còn giữ trong predictor (kèm bộ đếm id), để video sau không nhận track cũ."""
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker.reset()


def track_video(model, video_path: Path, video_id: str, options: dict) -> tuple[pd.DataFrame, dict]:
    """Chạy tracker trên 1 video, trả về (dòng thô, thống kê). frame_idx là số thứ tự frame đọc được, từ 0.

    persist=False: mỗi video một tracker mới; reset_trackers sau khi xong để chắc chắn không rò id.
    Detection chưa được gán track_id bị bỏ và đếm ở n_no_id.
    """
    rows, n_read, n_no_id = [], 0, 0
    results = model.track(source=str(video_path), classes=[PERSON_CLASS], stream=True, persist=False,
                          verbose=False, **options)
    try:
        for frame_idx, result in enumerate(results):
            n_read = frame_idx + 1
            boxes = result.boxes
            if boxes is None or len(boxes.xyxy) == 0:
                continue
            if boxes.id is None:
                n_no_id += len(boxes.xyxy)
                continue
            for track_id, corners, score in zip(_as_list(boxes.id), _as_list(boxes.xyxy), _as_list(boxes.conf)):
                rows.append((video_id, frame_idx, int(track_id), *map(float, corners), float(score)))
    finally:
        reset_trackers(model)
    return pd.DataFrame(rows, columns=TRACK_COLUMNS), {"n_read": n_read, "n_no_id": n_no_id}


def process_video(model, row: dict, root: Path, options: dict) -> tuple[pd.DataFrame | None, dict]:
    """Tracking + làm sạch 1 video theo metadata. Trả về (bảng hoặc None nếu lỗi, thống kê có 'warning'/'error')."""
    started = time.time()
    video_id, n_frames = row["video_id"], int(row["n_frames"])
    stats = {"video_id": video_id, "dataset": row["dataset"], "n_frames_meta": n_frames, "warning": None, "error": None}
    path = root / row["path"]
    if not path.is_file():
        return None, {**stats, "error": f"không thấy video {path}"}
    raw, read_stats = track_video(model, path, video_id, options)
    stats.update(read_stats)
    diff = stats["n_read"] - n_frames
    if diff:
        text = f"{video_id}: đọc được {stats['n_read']} frame, metadata {n_frames}"
        if abs(diff) > MAX_FRAME_DIFF:
            return None, {**stats, "error": text}
        stats["warning"] = text
    in_range = raw["frame_idx"] < n_frames
    stats["n_out_of_range"] = int((~in_range).sum())
    tracks, stats["n_zero_area"] = clip_to_frame(raw[in_range], int(row["width"]), int(row["height"]))
    tracks = cast_tracks(tracks) if len(tracks) else empty_tracks()
    stats.update(n_rows=len(tracks), n_tracks=int(tracks["track_id"].nunique()), seconds=round(time.time() - started, 2))
    return tracks, stats


def _shard_paths(out_dir: Path, video_id: str) -> tuple[Path, Path]:
    shard_dir = out_dir / "shards"
    return shard_dir / f"{video_id}.parquet", shard_dir / f"{video_id}.json"


def write_shard(out_dir: Path, tracks: pd.DataFrame, stats: dict) -> None:
    """Ghi parquet rồi mới ghi json (json là dấu hoàn tất); ghi file tạm rồi đổi tên để không để lại shard dở."""
    parquet, info = _shard_paths(out_dir, stats["video_id"])
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(parquet.with_suffix(".tmp"), index=False)
    os.replace(parquet.with_suffix(".tmp"), parquet)
    info.with_suffix(".tmp").write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    os.replace(info.with_suffix(".tmp"), info)


def read_shard(out_dir: Path, video_id: str) -> tuple[pd.DataFrame, dict] | None:
    """Shard hợp lệ (đủ 2 file, đọc được, đúng cột, đúng video, đúng số dòng) thì trả về, ngược lại None."""
    parquet, info = _shard_paths(out_dir, video_id)
    try:
        stats = json.loads(info.read_text(encoding="utf-8"))
        tracks = pd.read_parquet(parquet)
    except (OSError, ValueError):
        return None
    valid = (stats.get("video_id") == video_id and list(tracks.columns) == TRACK_COLUMNS
             and len(tracks) == stats.get("n_rows") and (tracks["video_id"] == video_id).all())
    return (tracks, stats) if valid else None
