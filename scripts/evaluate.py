"""Đánh giá predictions.csv trên một tập, in bảng và lưu JSON vào reports/results/.

Đầu vào: --pred, --meta, --split (train|val|test|all), --dataset (urfd|le2i).
Đầu ra: reports/results/<tag>_<dataset>_<split>.json, có một kết quả cho mỗi giá trị --T.
--split all dùng toàn bộ video có nhãn của bộ dữ liệu, cho cross-dataset.
Ví dụ: python scripts/evaluate.py --pred predictions.csv --meta metadata.csv --split test --dataset urfd --T 0.5 1.0 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.common import METADATA_DTYPES  # noqa: E402
from src.eval.event_metrics import (DEFAULT_MERGE_GAP_S, DEFAULT_THR, check_predictions,  # noqa: E402
                                    evaluate)


def select_videos(meta: pd.DataFrame, splits_path: Path, dataset: str, split: str) -> pd.DataFrame:
    """Lấy các dòng metadata của tập cần đánh giá; chỉ video has_event_label = 1."""
    labeled = meta[(meta["dataset"] == dataset) & (meta["has_event_label"] == 1)]
    if split == "all":
        return labeled
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    return labeled[labeled["video_id"].isin(splits[dataset][split])]


def format_value(value) -> str:
    """Định dạng 1 ô: None thành 'n/a', số thực 3 chữ số."""
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_table(results: list[dict]) -> None:
    """In bảng độ đo sự kiện, frame và ADL negative khó cho từng T."""
    for result in results:
        event, frame, hard = result["event"], result["frame"], result["hard_negative"]
        print(f"\nT = {result['params']['tolerance_s']} s | thr = {result['params']['thr']}")
        for key in ("n_videos", "n_events", "tp", "fp", "fn", "ignored_alarms", "precision", "recall", "f1",
                    "far_per_hour", "far_nonfall_per_hour", "latency_mean_s", "latency_median_s"):
            print(f"  {key:22s} {format_value(event[key])}")
        print(f"  frame: accuracy={format_value(frame['accuracy'])} roc_auc={format_value(frame['roc_auc'])} "
              f"pr_auc={format_value(frame['pr_auc'])} (trên {frame['n_frames']} frame)")
        print(f"  ADL negative khó: {hard['n_videos']} video | có FP: {hard['n_videos_with_fp']} "
              f"| FP: {hard['fp']} | FAR: {format_value(hard['far_per_hour'])}/giờ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá mức sự kiện và mức frame.")
    parser.add_argument("--pred", required=True, help="predictions.csv")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "all"], help="tập đánh giá")
    parser.add_argument("--dataset", required=True, choices=["urfd", "le2i"], help="bộ dữ liệu")
    parser.add_argument("--splits", help="splits.json (mặc định: cạnh metadata.csv)")
    parser.add_argument("--frame-labels", help="urfd_frame_labels.csv (mặc định: annotations/ cạnh metadata.csv)")
    parser.add_argument("--thr", type=float, default=DEFAULT_THR, help="ngưỡng fall_score")
    parser.add_argument("--T", type=float, nargs="+", default=[1.0], help="dung sai T (giây), nhiều giá trị được")
    parser.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP_S, help="merge_gap (giây)")
    parser.add_argument("--out-dir", default="reports/results", help="thư mục lưu JSON")
    parser.add_argument("--tag", help="tiền tố tên file JSON (mặc định: tên file --pred)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = Path(args.meta)
    meta = pd.read_csv(meta_path, dtype=METADATA_DTYPES)
    predictions = pd.read_csv(args.pred)
    check_predictions(predictions, meta)
    videos = select_videos(meta, Path(args.splits) if args.splits else meta_path.parent / "splits.json",
                           args.dataset, args.split)
    labels_path = Path(args.frame_labels) if args.frame_labels else meta_path.parent / "annotations" / "urfd_frame_labels.csv"
    frame_labels = pd.read_csv(labels_path) if args.dataset == "urfd" and labels_path.exists() else None
    in_scope = predictions[predictions["video_id"].isin(videos["video_id"])]
    results = [evaluate(in_scope, videos, args.thr, tolerance, args.merge_gap, frame_labels) for tolerance in args.T]
    print_table(results)
    tag = args.tag or Path(args.pred).stem
    out_path = Path(args.out_dir) / f"{tag}_{args.dataset}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pred": str(args.pred), "dataset": args.dataset, "split": args.split, "results": results}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    n_skipped = predictions["video_id"].nunique() - in_scope["video_id"].nunique()
    print(f"\nTÓM TẮT | video đánh giá: {len(videos)} | video có dự đoán ngoài tập, bỏ qua: {n_skipped} | Đã ghi: {out_path}")


if __name__ == "__main__":
    main()
