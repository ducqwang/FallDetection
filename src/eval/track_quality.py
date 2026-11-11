"""Chất lượng detector + tracker, tách lỗi do detector khỏi lỗi do luật M1.

Le2i: so track chính (select_main_track, chọn theo IoU với GT) với GT bbox đã dedupe. URFD không có GT bbox nên chỉ
đo số track và tỉ lệ frame có người. Mọi frame là frame_idx 0-based, hộp pixel xyxy.
Lưu ý: track chính Le2i được chọn bằng chính GT nên số đo là cận trên của những gì M1 nhận được.
Ví dụ: per_video = le2i_track_quality(tracks, meta, gt); summarize(per_video, "scene")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.common import drop_duplicate_frames
from ..data.labels import bbox_iou, select_main_track

IOU_HIT = 0.5
BOX = ["x1", "y1", "x2", "y2"]


def _main_track_ious(track: pd.DataFrame, gt: pd.DataFrame, width: int, height: int) -> np.ndarray:
    """IoU của track chính trên từng frame có GT; frame track thiếu tính IoU = 0."""
    joined = gt[["frame_idx", *BOX]].merge(track[["frame_idx", *BOX]], on="frame_idx", how="left",
                                           suffixes=("_gt", "_tr"))
    have = joined["x1_tr"].notna().to_numpy()
    ious = np.zeros(len(joined))
    if have.any():
        boxes = joined.loc[have, [f"{c}_tr" for c in BOX]].to_numpy(dtype=float)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        ious[have] = bbox_iou(boxes, joined.loc[have, [f"{c}_gt" for c in BOX]].to_numpy(dtype=float))
    return ious


def le2i_track_quality(tracks: pd.DataFrame, meta: pd.DataFrame, gt_bbox: pd.DataFrame) -> pd.DataFrame:
    """Mỗi video Le2i một dòng: số track, track chính, số frame có GT, tỉ lệ frame GT track chính có hộp,
    IoU trung bình (mean_iou) và tỉ lệ frame GT có IoU >= 0.5 (hit_rate). Video không có GT: NaN."""
    gt_all, _ = drop_duplicate_frames(gt_bbox)
    gt_by_video = dict(tuple(gt_all.groupby("video_id")))
    tracks_by_video = dict(tuple(tracks.groupby("video_id")))
    rows = []
    for row in meta[meta["dataset"] == "le2i"].to_dict("records"):
        video_id = row["video_id"]
        own = tracks_by_video.get(video_id, tracks.iloc[0:0])
        gt = gt_by_video.get(video_id, gt_all.iloc[0:0])
        main = select_main_track(own, row, gt)
        record = {"video_id": video_id, "scene": row["scene"], "label": row["label"],
                  "n_tracks": int(own["track_id"].nunique()), "main_track": main, "n_gt_frames": len(gt),
                  "main_coverage": np.nan, "mean_iou": np.nan, "hit_rate": np.nan}
        if len(gt):
            track = own[own["track_id"] == main] if main is not None else own.iloc[0:0]
            ious = _main_track_ious(track, gt, int(row["width"]), int(row["height"]))
            record.update(main_coverage=float(gt["frame_idx"].isin(track["frame_idx"]).mean()),
                          mean_iou=float(ious.mean()), hit_rate=float((ious >= IOU_HIT).mean()))
        rows.append(record)
    return pd.DataFrame(rows)


def urfd_track_quality(tracks: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mỗi video URFD một dòng: số track và tỉ lệ frame có ít nhất một track."""
    rows = []
    tracks_by_video = dict(tuple(tracks.groupby("video_id")))
    for row in meta[meta["dataset"] == "urfd"].to_dict("records"):
        own = tracks_by_video.get(row["video_id"], tracks.iloc[0:0])
        rows.append({"video_id": row["video_id"], "label": row["label"], "n_frames": int(row["n_frames"]),
                     "n_tracks": int(own["track_id"].nunique()),
                     "frac_frames_with_track": own["frame_idx"].nunique() / int(row["n_frames"])})
    return pd.DataFrame(rows)


def summarize(per_video: pd.DataFrame, by: str) -> pd.DataFrame:
    """Trung bình theo nhóm (scene hoặc label) cộng dòng 'all'; trung bình đều theo video, bỏ video NaN."""
    metrics = [c for c in ("n_tracks", "main_coverage", "mean_iou", "hit_rate", "frac_frames_with_track")
               if c in per_video]
    grouped = per_video.groupby(by)[metrics].mean()
    grouped.insert(0, "n_videos", per_video.groupby(by).size())
    total = per_video[metrics].mean().to_frame("all").T
    total.insert(0, "n_videos", len(per_video))
    return pd.concat([grouped, total]).rename_axis(by).reset_index()
