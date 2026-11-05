"""Gom các JSON của scripts/evaluate.py thành một bảng kết quả sơ bộ.

Đầu vào: thư mục reports/results. Đầu ra: file CSV và bảng in ra màn hình.
Mỗi dòng là một tổ hợp file JSON và dung sai T.
Ví dụ: python scripts/summarize_results.py --results-dir reports/results --out reports/results/m1_summary.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MAIN_FAR_COLUMN = "far_nonfall_per_hour (CHÍNH)"
SECONDARY_FAR_COLUMN = "far_per_hour (phụ)"
EVENT_KEYS = ["n_videos", "n_events", "tp", "fp", "fn", "precision", "recall", "f1", "far_nonfall_per_hour",
              "far_per_hour", "latency_mean_s", "latency_median_s"]
ROUND_DECIMALS = 3


def result_rows(path: Path) -> list[dict]:
    """Các dòng bảng từ 1 file JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in payload["results"]:
        hard = result["hard_negative"]
        rows.append({"run": path.stem, "T": result["params"]["tolerance_s"], "thr": result["params"]["thr"],
                     **{key: result["event"][key] for key in EVENT_KEYS}, "roc_auc": result["frame"]["roc_auc"],
                     "pr_auc": result["frame"]["pr_auc"], "hard_neg_videos": hard["n_videos"],
                     "hard_neg_videos_with_fp": hard["n_videos_with_fp"], "hard_neg_fp": hard["fp"]})
    return rows


def main() -> None:
    """Đọc mọi JSON trong thư mục kết quả và ghi bảng CSV."""
    parser = argparse.ArgumentParser(description="Gom JSON đánh giá thành bảng CSV.")
    parser.add_argument("--results-dir", default="reports/results", help="thư mục chứa JSON")
    parser.add_argument("--out", required=True, help="file CSV sẽ ghi")
    args = parser.parse_args()
    paths = sorted(Path(args.results_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"Không có file JSON trong {args.results_dir}")
    table = pd.DataFrame([row for path in paths for row in result_rows(path)]).round(ROUND_DECIMALS)
    table = table.rename(columns={"far_nonfall_per_hour": MAIN_FAR_COLUMN, "far_per_hour": SECONDARY_FAR_COLUMN})
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"TÓM TẮT | file JSON: {len(paths)} | dòng: {len(table)} | bỏ qua: 0 | Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
