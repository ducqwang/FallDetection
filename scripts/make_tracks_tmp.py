# TẠM THỜI — sẽ thay bằng output chính thức của TV2
"""Sinh tracks.parquet tạm bằng YOLO11 + ByteTrack để M1 chạy được khi chưa có cache của TV2.

Đầu vào: metadata.csv và thư mục chứa videos/*.mp4. Đầu ra: tracks.parquet đúng schema.
Cần GPU và Internet (lần đầu tải trọng số YOLO11). Chỉ chạy trên mp4 đã chuẩn hóa, không đọc .avi.
Video has_event_label = 0 được bỏ qua.
Ví dụ: python scripts/make_tracks_tmp.py --meta metadata.csv --out tracks.parquet --dataset urfd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.common import METADATA_DTYPES, TRACK_COLUMNS  # noqa: E402

PERSON_CLASS = 0
TRACKER = "bytetrack.yaml"


def track_video(model: "YOLO", video_path: Path, video_id: str, conf: float, imgsz: int) -> list[dict]:
    """Chạy YOLO11 + ByteTrack trên 1 video; frame_idx là thứ tự frame, 0-based; conf ∈ [0, 1]; imgsz tính bằng pixel.

    Mỗi lời gọi tạo tracker mới (persist=False) để track_id không lẫn giữa các video.
    """
    rows: list[dict] = []
    results = model.track(source=str(video_path), tracker=TRACKER, classes=[PERSON_CLASS], conf=conf,
                          imgsz=imgsz, stream=True, persist=False, verbose=False)
    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        for track_id, corners, score in zip(ids, boxes.xyxy.cpu().tolist(), scores):
            x1, y1, x2, y2 = corners
            rows.append({"video_id": video_id, "frame_idx": frame_idx, "track_id": track_id,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "det_conf": score})
    return rows


def main() -> None:
    """Chạy tracking trên các video đã chọn và ghi tracks.parquet."""
    parser = argparse.ArgumentParser(description="TẠM THỜI: sinh tracks.parquet bằng YOLO11 + ByteTrack.")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--out", required=True, help="tracks.parquet sẽ ghi")
    parser.add_argument("--videos-root", help="thư mục chứa videos/ (mặc định: cạnh metadata.csv)")
    parser.add_argument("--dataset", choices=["urfd", "le2i"], help="chỉ chạy 1 bộ (mặc định: cả hai)")
    parser.add_argument("--model", default="yolo11n.pt", help="trọng số YOLO11")
    parser.add_argument("--conf", type=float, default=0.25, help="ngưỡng phát hiện")
    parser.add_argument("--imgsz", type=int, default=640, help="kích thước ảnh đầu vào YOLO")
    args = parser.parse_args()
    from ultralytics import YOLO

    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    root = Path(args.videos_root) if args.videos_root else Path(args.meta).parent
    videos = meta[meta["has_event_label"] == 1]
    if args.dataset:
        videos = videos[videos["dataset"] == args.dataset]
    model = YOLO(args.model)
    rows: list[dict] = []
    n_empty = 0
    for index, row in enumerate(videos.itertuples(), start=1):
        video_rows = track_video(model, root / row.path, row.video_id, args.conf, args.imgsz)
        n_empty += not video_rows
        rows += video_rows
        print(f"[{index}/{len(videos)}] {row.video_id}: {len(video_rows)} dòng track")
    tracks = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(args.out, index=False)
    print(f"TÓM TẮT | video đã chạy: {len(videos)} | bỏ qua: {len(meta) - len(videos)} (has_event_label = 0 hoặc khác --dataset) "
          f"| không có track: {n_empty} | dòng: {len(tracks)} | Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
