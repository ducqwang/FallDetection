"""Báo cáo chất lượng detector của tracks.parquet, để tách lỗi do detector khỏi lỗi do luật M1.

Le2i (có GT bbox): mỗi video IoU trung bình của track chính so với GT và tỉ lệ frame GT có IoU >= 0.5;
tổng hợp theo scene và theo lớp fall/ADL. URFD: số track trung bình mỗi video, tỉ lệ frame có ít nhất một track.
Đầu vào: tracks.parquet, metadata.csv, le2i_gt_bbox.csv (dedupe bằng drop_duplicate_frames).
Đầu ra (--out-dir, mặc định <thư mục tracks>/quality): 5 CSV, in bảng tóm tắt.
Ví dụ: python scripts/check_tracks_quality.py --tracks /kaggle/working/tracks/tracks.parquet --root $ROOT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.common import METADATA_DTYPES  # noqa: E402
from src.data.loader import load_tracks  # noqa: E402
from src.eval.track_quality import le2i_track_quality, summarize, urfd_track_quality  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chất lượng detector/tracker trên Le2i (so GT) và URFD.")
    parser.add_argument("--tracks", required=True, help="tracks.parquet")
    parser.add_argument("--root", help="thư mục dataset (mặc định: cạnh --meta)")
    parser.add_argument("--meta", help="metadata.csv (mặc định: <root>/metadata.csv)")
    parser.add_argument("--gt-bbox", help="le2i_gt_bbox.csv (mặc định: <root>/annotations/le2i_gt_bbox.csv)")
    parser.add_argument("--out-dir", help="thư mục ghi CSV (mặc định: <thư mục tracks>/quality)")
    args = parser.parse_args(argv)
    if not (args.root or args.meta):
        parser.error("cần --root hoặc --meta")
    return args


def processed_videos(tracks_path: Path, tracks: pd.DataFrame, meta: pd.DataFrame) -> set[str]:
    """Video đã chạy tracking: lấy từ tracks_meta.json cạnh tracks (gồm cả video không có người);
    không có sidecar thì lấy mọi video has_event_label = 1 thuộc các bộ có mặt trong tracks."""
    sidecar = tracks_path.parent / "tracks_meta.json"
    if sidecar.is_file():
        return {item["video_id"] for item in json.loads(sidecar.read_text(encoding="utf-8"))["videos"]}
    datasets = set(meta.loc[meta["video_id"].isin(set(tracks["video_id"])), "dataset"])
    return set(meta.loc[(meta["has_event_label"] == 1) & meta["dataset"].isin(datasets), "video_id"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root) if args.root else Path(args.meta).parent
    meta = pd.read_csv(args.meta or root / "metadata.csv", dtype=METADATA_DTYPES)
    tracks = load_tracks(args.tracks)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.tracks).parent / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    written = []
    meta = meta[meta["video_id"].isin(processed_videos(Path(args.tracks), tracks, meta))]
    print(f"Đánh giá {len(meta)} video ({meta['dataset'].value_counts().to_dict()})")
    if (meta["dataset"] == "le2i").any():
        gt = pd.read_csv(args.gt_bbox or root / "annotations" / "le2i_gt_bbox.csv")
        per_video = le2i_track_quality(tracks, meta, gt)
        tables = {"le2i_track_quality.csv": per_video, "le2i_track_quality_by_scene.csv": summarize(per_video, "scene"),
                  "le2i_track_quality_by_label.csv": summarize(per_video, "label")}
        for name, table in tables.items():
            table.to_csv(out_dir / name, index=False)
            written.append(out_dir / name)
        print("LE2I | track chính so với GT (mean_iou, hit_rate = tỉ lệ frame GT có IoU >= 0.5)")
        print(tables["le2i_track_quality_by_scene.csv"].round(3).to_string(index=False))
        print(tables["le2i_track_quality_by_label.csv"].round(3).to_string(index=False))
        worst = per_video.sort_values("hit_rate").head(10)
        print("10 video Le2i hit_rate thấp nhất:", worst[["video_id", "scene", "label", "n_tracks", "mean_iou",
                                                         "hit_rate"]].round(3).to_string(index=False), sep="\n")
    if (meta["dataset"] == "urfd").any():
        per_video = urfd_track_quality(tracks, meta)
        summary = summarize(per_video, "label")
        per_video.to_csv(out_dir / "urfd_track_quality.csv", index=False)
        summary.to_csv(out_dir / "urfd_track_quality_by_label.csv", index=False)
        written += [out_dir / "urfd_track_quality.csv", out_dir / "urfd_track_quality_by_label.csv"]
        print("URFD | số track trung bình mỗi video, tỉ lệ frame có ít nhất một track")
        print(summary.round(3).to_string(index=False))
    print("Đã ghi:", *written, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
