"""Grid search ngưỡng M1 CHỈ trên tập val, tối ưu theo event-F1. Không bao giờ đọc tập test.

Đầu vào: --tracks hoặc --gt-bbox, --meta, --dataset, lưới trong configs/m1.yaml.
Đầu ra: yaml ngưỡng tốt nhất (--out) và csv toàn bộ lưới cạnh đó.
Tiêu chí: F1 cao nhất; hòa thì ít FP hơn; vẫn hòa thì lấy tổ hợp đứng trước trong lưới.
Ví dụ: python scripts/tune_m1.py --tracks tracks.parquet --meta metadata.csv --dataset urfd --out reports/results/m1_tuned_urfd.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.common import METADATA_DTYPES  # noqa: E402
from src.eval.event_metrics import evaluate  # noqa: E402
from src.models import m1_rule  # noqa: E402

ALARM_KEYS = ("thr",)


def rule_combinations(grid: dict) -> list[dict]:
    """Mọi tổ hợp tham số luật trong lưới (trừ thr)."""
    names = [name for name in grid if name not in ALARM_KEYS]
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def search(items: list[dict], videos: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Chấm toàn bộ lưới trên các video val. Trả về bảng kết quả, mỗi tổ hợp 1 dòng."""
    alarm = config["alarm"]
    rows = []
    for combo in rule_combinations(config["grid"]):
        rules = {**config["rules"], **combo}
        predictions = m1_rule.predictions_from_signals(items, rules, config["weights"])
        for thr in config["grid"]["thr"]:
            event = evaluate(predictions, videos, thr, alarm["tolerance_s"], alarm["merge_gap_s"])["event"]
            rows.append({**combo, "thr": thr, "f1": event["f1"], "precision": event["precision"],
                         "recall": event["recall"], "fp": event["fp"], "latency_median_s": event["latency_median_s"]})
    return pd.DataFrame(rows)


def best_row(table: pd.DataFrame) -> pd.Series:
    """Dòng tốt nhất: F1 giảm dần, FP tăng dần, giữ thứ tự lưới khi hòa."""
    return table.sort_values(["f1", "fp"], ascending=[False, True], kind="mergesort").iloc[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune ngưỡng M1 chỉ trên tập val.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracks", help="tracks.parquet")
    source.add_argument("--gt-bbox", help="le2i_gt_bbox.csv (tune cho chế độ GT)")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--splits", help="splits.json (mặc định: cạnh metadata.csv)")
    parser.add_argument("--dataset", required=True, choices=["urfd", "le2i"], help="bộ dữ liệu")
    parser.add_argument("--config", default="configs/m1.yaml", help="file cấu hình M1")
    parser.add_argument("--out", required=True, help="yaml ngưỡng tốt nhất sẽ ghi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = m1_rule.load_config(args.config)
    meta_path = Path(args.meta)
    meta = pd.read_csv(meta_path, dtype=METADATA_DTYPES)
    splits_path = Path(args.splits) if args.splits else meta_path.parent / "splits.json"
    val_ids = set(json.loads(splits_path.read_text(encoding="utf-8"))[args.dataset]["val"])
    videos = meta[meta["video_id"].isin(val_ids)]
    tracks = m1_rule.gt_boxes_as_tracks(pd.read_csv(args.gt_bbox)) if args.gt_bbox else pd.read_parquet(args.tracks)
    # Chỉ giữ track của val ngay từ đầu để tập test không thể ảnh hưởng tới kết quả.
    val_tracks = tracks[tracks["video_id"].isin(val_ids)]
    table = search(m1_rule.signals_by_track(val_tracks, meta, config), videos, config)
    best = best_row(table)
    names = [name for name in config["grid"] if name not in ALARM_KEYS]
    tuned = {"rules": {name: float(best[name]) for name in names}, "alarm": {"thr": float(best["thr"])},
             "val": {"dataset": args.dataset, "n_videos": len(videos), "f1": float(best["f1"]),
                     "precision": float(best["precision"]), "recall": float(best["recall"]), "fp": int(best["fp"])}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(tuned, allow_unicode=True, sort_keys=False), encoding="utf-8")
    grid_path = out_path.with_name(f"{out_path.stem}_grid.csv")
    table.to_csv(grid_path, index=False)
    print(f"TÓM TẮT | val: {len(videos)} video | tổ hợp đã thử: {len(table)} | F1 tốt nhất: {best['f1']:.3f} "
          f"(P={best['precision']:.3f}, R={best['recall']:.3f}, FP={int(best['fp'])}) | Đã ghi: {out_path}, {grid_path}")


if __name__ == "__main__":
    main()
