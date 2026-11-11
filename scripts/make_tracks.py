"""Sinh tracks.parquet chính thức của TV1 bằng YOLO11 + ByteTrack (chỉ lớp người), chạy tiếp được khi Kaggle ngắt.

Đầu vào: metadata.csv và thư mục dataset chứa videos/*.mp4 (cột path). Đầu ra trong --out-dir:
  shards/<video_id>.parquet + .json   mỗi video một shard, ghi ngay khi xong (--resume bỏ qua shard hợp lệ)
  tracks.parquet                      ghép mọi shard, sắp theo (video_id, frame_idx, track_id), đã qua validate_tracks
  tracks_meta.json                    tham số, phiên bản mô hình/thư viện, tracker yaml, git commit, số dòng bỏ, cảnh báo
Schema: video_id, frame_idx (0-based, thứ tự frame đọc được), track_id (int), x1, y1, x2, y2 (pixel, float), det_conf.
Cần GPU và Internet (lần đầu tải trọng số). Thoát mã 1 nếu có video lỗi, 2 nếu tracks vi phạm schema.
Ví dụ: python scripts/make_tracks.py --root /kaggle/input/nhom05-falldet-processed --limit 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.common import METADATA_DTYPES  # noqa: E402
from src.data.track_runner import process_video, read_shard, write_shard  # noqa: E402
from src.data.tracks import empty_tracks, validate_tracks  # noqa: E402

DROP_COUNTERS = ("n_no_id", "n_zero_area", "n_out_of_range")
KAGGLE_WORKING = Path("/kaggle/working")


def default_out_dir() -> Path:
    """Một chỗ duy nhất: /kaggle/working/tracks trên Kaggle, reports/tracks của repo khi chạy máy khác."""
    return KAGGLE_WORKING / "tracks" if KAGGLE_WORKING.is_dir() else REPO / "reports" / "tracks"


def auto_device() -> str:
    """GPU đầu tiên nếu có, ngược lại CPU."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def tracker_yaml_text(tracker: str) -> str | None:
    """Nội dung tracker yaml: file cục bộ, hoặc file cấu hình đi kèm ultralytics."""
    if Path(tracker).is_file():
        return Path(tracker).read_text(encoding="utf-8")
    try:
        from ultralytics.utils.checks import check_yaml
        return Path(check_yaml(tracker)).read_text(encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - chỉ để ghi sidecar, không chặn kết quả
        return f"không đọc được: {error}"


def _version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "chưa cài"


def environment_info(model, model_name: str) -> dict:
    """Tên, file và sha256 trọng số, phiên bản thư viện, git commit của repo."""
    weights = Path(getattr(model, "ckpt_path", None) or model_name)
    try:
        import cv2
        opencv = cv2.__version__
    except ImportError:
        opencv = "chưa cài"

    def git(*cmd: str) -> str:
        done = subprocess.run(["git", "-C", str(REPO), *cmd], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else "không rõ"
    return {"model": model_name, "weights_file": str(weights),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else None,
            "ultralytics": _version("ultralytics"), "torch": _version("torch"), "opencv": opencv,
            "python": sys.version.split()[0], "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": git("status", "--porcelain") not in ("", "không rõ")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh tracks.parquet chính thức (YOLO11 + ByteTrack, lớp người).")
    parser.add_argument("--meta", help="metadata.csv (mặc định: <root>/metadata.csv)")
    parser.add_argument("--root", help="thư mục dataset để ghép cột path (mặc định: cạnh --meta, "
                                       "hoặc tự tìm nhom05-falldet-processed)")
    parser.add_argument("--out-dir", help=f"thư mục output (mặc định: {default_out_dir()})")
    parser.add_argument("--dataset", choices=["all", "urfd", "le2i"], default="all")
    parser.add_argument("--model", default="yolo11m.pt", help="tên trọng số YOLO11 hoặc đường dẫn file cục bộ")
    parser.add_argument("--conf", type=float, default=0.25, help="ngưỡng phát hiện")
    parser.add_argument("--imgsz", type=int, default=640, help="kích thước ảnh đầu vào YOLO (pixel)")
    parser.add_argument("--iou", type=float, default=0.7, help="ngưỡng IoU của NMS")
    parser.add_argument("--device", help="'0', 'cpu', ... (mặc định: GPU nếu có)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="tracker yaml của ultralytics hoặc file cục bộ")
    parser.add_argument("--limit", type=int, help="chỉ chạy N video đầu (chạy thử)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="bỏ qua video đã có shard hợp lệ (mặc định bật; --no-resume để chạy lại hết)")
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """(metadata.csv, thư mục dataset, thư mục output) từ tham số."""
    if args.root:
        root = Path(args.root)
    elif args.meta:
        root = Path(args.meta).parent
    else:
        from src.data.loader import data_root
        root = data_root()
    meta_path = Path(args.meta) if args.meta else root / "metadata.csv"
    return meta_path, root, Path(args.out_dir) if args.out_dir else default_out_dir()


def select_videos(meta: pd.DataFrame, dataset: str, limit: int | None) -> pd.DataFrame:
    """Video has_event_label = 1 của bộ đã chọn, theo thứ tự metadata; limit lấy N video đầu."""
    videos = meta[meta["has_event_label"] == 1]
    if dataset != "all":
        videos = videos[videos["dataset"] == dataset]
    return videos.head(limit) if limit else videos


def load_model(name: str):
    from ultralytics import YOLO
    return YOLO(name)


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes // 60:d}:{minutes % 60:02d}:{secs:02d}"


def track_all(model, videos: pd.DataFrame, root: Path, out_dir: Path, options: dict, resume: bool,
              model_name: str) -> list[dict]:
    """Chạy từng video, ghi shard ngay khi xong, in tiến độ. Trả về thống kê mọi video (kể cả lấy từ shard cũ).

    Shard cũ chỉ được dùng lại khi cùng model và cùng conf/iou/imgsz/tracker; khác thì chạy lại video đó.
    """
    signature = {"model": model_name, **{key: value for key, value in options.items() if key != "device"}}
    all_stats, started, n_done_now = [], time.time(), 0
    rows = videos.to_dict("records")
    for index, row in enumerate(rows, start=1):
        cached = read_shard(out_dir, row["video_id"]) if resume else None
        if cached is not None and cached[1].get("options") == signature:
            all_stats.append({**cached[1], "resumed": True})
            print(f"[{index}/{len(rows)}] {row['video_id']}: đã có shard, bỏ qua")
            continue
        if cached is not None:
            print(f"[{index}/{len(rows)}] {row['video_id']}: shard cũ khác tham số {cached[1].get('options')}, chạy lại")
        tracks, stats = process_video(model, row, root, options)
        stats["options"] = signature
        all_stats.append({**stats, "resumed": False})
        if tracks is None:
            print(f"[{index}/{len(rows)}] {row['video_id']}: LỖI {stats['error']}")
            continue
        write_shard(out_dir, tracks, stats)
        n_done_now += 1
        elapsed = time.time() - started
        left = (len(rows) - index) * elapsed / n_done_now
        print(f"[{index}/{len(rows)}] {row['video_id']}: {stats['n_rows']} dòng, {stats['n_tracks']} track, "
              f"{stats['n_read']} frame, {stats['seconds']:.1f}s | đã chạy {_clock(elapsed)}, còn ~{_clock(left)}")
    return all_stats


def merge_shards(out_dir: Path, video_ids: list[str]) -> pd.DataFrame:
    """Ghép shard của các video đã chọn, sắp theo (video_id, frame_idx, track_id)."""
    parts = [read_shard(out_dir, video_id)[0] for video_id in video_ids]
    if not parts:
        return empty_tracks()
    tracks = pd.concat([part for part in parts if len(part)] or parts[:1], ignore_index=True)
    return tracks.sort_values(["video_id", "frame_idx", "track_id"], kind="stable").reset_index(drop=True)


def write_sidecar(path: Path, args: argparse.Namespace, env: dict, stats: list[dict], summary: dict) -> None:
    sidecar = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               **env, "tracker_yaml": tracker_yaml_text(args.tracker), **summary, "videos": stats}
    path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace, model=None) -> int:
    """Toàn bộ quy trình; model=None thì tải YOLO theo --model. Trả về mã thoát."""
    started = time.time()
    meta_path, root, out_dir = resolve_inputs(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"metadata: {meta_path}\ndataset: {root}\nOUTPUT: {out_dir.resolve()}")
    meta = pd.read_csv(meta_path, dtype=METADATA_DTYPES)
    videos = select_videos(meta, args.dataset, args.limit)
    args.device = args.device or auto_device()
    model = model if model is not None else load_model(args.model)
    options = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "device": args.device, "tracker": args.tracker}
    # shard chỉ dùng lại khi cùng mô hình và tham số tracking (device không đổi kết quả nên không so).
    stats = track_all(model, videos, root, out_dir, options, args.resume, args.model)
    errors = [item["error"] for item in stats if item.get("error")]
    summary = {"n_videos": len(videos), "n_skipped_meta": len(meta) - len(videos),
               "dropped": {key: sum(item.get(key, 0) for item in stats) for key in DROP_COUNTERS},
               "warnings": [item["warning"] for item in stats if item.get("warning")], "errors": errors,
               "videos_without_tracks": [item["video_id"] for item in stats if item.get("n_rows") == 0],
               "run_seconds": None, "tracking_seconds_all_videos": round(sum(item.get("seconds", 0) for item in stats), 1)}
    code, tracks_path = 0, out_dir / "tracks.parquet"
    if errors:
        code = 1
        print(f"LỖI | {len(errors)} video lỗi, KHÔNG ghi tracks.parquet (shard của video tốt vẫn giữ):", *errors, sep="\n  ")
    else:
        tracks = merge_shards(out_dir, videos["video_id"].tolist())
        check = validate_tracks(tracks, videos)
        summary.update(n_rows=len(tracks), n_tracks=int(tracks[["video_id", "track_id"]].drop_duplicates().shape[0]),
                       validation=[f"[{c}] {t}" for c, t in check.issues])
        print("validate_tracks:", check.summary(), sep="\n")
        if check.ok:
            tracks.to_parquet(tracks_path, index=False)
        else:
            code = 2
            print("LỖI | tracks vi phạm schema, KHÔNG ghi tracks.parquet")
    summary["run_seconds"] = round(time.time() - started, 1)
    write_sidecar(out_dir / "tracks_meta.json", args, environment_info(model, args.model), stats, summary)
    dropped = ", ".join(f"{key} {value}" for key, value in summary["dropped"].items())
    print(f"CẢNH BÁO số frame lệch metadata ({len(summary['warnings'])}):", *summary["warnings"], sep="\n  ")
    print(f"TÓM TẮT | video: {len(videos)} (bỏ qua {summary['n_skipped_meta']} do --dataset/--limit/has_event_label) "
          f"| không có người: {len(summary['videos_without_tracks'])} {summary['videos_without_tracks']} "
          f"| dòng: {summary.get('n_rows', '-')} | đã bỏ: {dropped} | thời gian: {_clock(summary['run_seconds'])}")
    print(f"Đã ghi: {tracks_path if code == 0 else '(không ghi tracks.parquet)'} | {out_dir / 'tracks_meta.json'}")
    return code


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
