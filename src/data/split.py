"""Việc 2: chia train/val/test 70/15/15 theo video, phân tầng, chống rò rỉ.

Đầu vào: metadata.csv. Đầu ra: splits.json theo docs/SCHEMA.md.
Chỉ video has_event_label = 1 được đưa vào. URFD phân tầng theo (nhãn, adl_hard_neg); Le2i theo (nhãn, scene).
Mọi camera của cùng một chuỗi vào cùng một tập (chia theo sequence_id).
Ví dụ: python -m src.data.split --meta metadata.csv --out splits.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .common import METADATA_DTYPES, sequence_id

DEFAULT_RATIO = (0.70, 0.15, 0.15)
PARTS = ("train", "val", "test")
MIN_LE2I_TEST_ADL = 3
MIN_URFD_TEST_HARD_NEG = 1


def _stratum(row: pd.Series) -> tuple[str, str]:
    """Khóa phân tầng của 1 video."""
    if row["dataset"] == "urfd":
        return (row["label"], f"hard_neg={int(row['adl_hard_neg'])}")
    return (row["label"], row["scene"])


def _part_sizes(n_groups: int, ratio: tuple[float, float, float]) -> tuple[int, int]:
    """Số nhóm vào train và val của 1 tầng; phần còn lại vào test."""
    n_train = round(n_groups * ratio[0])
    n_val = round(n_groups * ratio[1])
    return n_train, n_val


def split_dataset(meta: pd.DataFrame, seed: int, ratio: tuple[float, float, float]) -> dict[str, list[str]]:
    """Chia 1 bộ dữ liệu. Trả về {'train': [...], 'val': [...], 'test': [...]} đã sắp xếp."""
    usable = meta[meta["has_event_label"] == 1].copy()
    usable["sequence"] = usable["video_id"].map(sequence_id)
    strata: dict[tuple[str, str], list[str]] = {}
    for sequence, group in usable.groupby("sequence", sort=True):
        keys = {_stratum(row) for _, row in group.iterrows()}
        if len(keys) != 1:
            raise ValueError(f"Các camera của {sequence} khác tầng: {keys}")
        strata.setdefault(keys.pop(), []).append(sequence)
    rng = random.Random(seed)
    chosen = {part: [] for part in PARTS}
    for key in sorted(strata):  # duyệt theo thứ tự cố định để kết quả tái lập được
        sequences = strata[key]
        rng.shuffle(sequences)
        n_train, n_val = _part_sizes(len(sequences), ratio)
        chosen["train"] += sequences[:n_train]
        chosen["val"] += sequences[n_train:n_train + n_val]
        chosen["test"] += sequences[n_train + n_val:]
    video_of = usable.groupby("sequence")["video_id"].apply(list).to_dict()
    return {part: sorted(v for s in chosen[part] for v in video_of[s]) for part in PARTS}


def validate_splits(splits: dict, meta: pd.DataFrame) -> None:
    """Raise nếu vi phạm: trùng tập, rò rỉ chuỗi, thiếu lớp, thiếu negative khó hoặc ADL ở test."""
    indexed = meta.set_index("video_id")
    for dataset in ("urfd", "le2i"):
        parts = splits[dataset]
        all_ids = [v for part in PARTS for v in parts[part]]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError(f"{dataset}: có video nằm ở 2 tập")
        sequence_part = {}
        for part in PARTS:
            for video in parts[part]:
                if sequence_part.setdefault(sequence_id(video), part) != part:
                    raise ValueError(f"{dataset}: rò rỉ chuỗi {sequence_id(video)}")
            labels = set(indexed.loc[parts[part], "label"])
            if labels != {"fall", "adl"}:
                raise ValueError(f"{dataset}/{part}: phải có cả fall và adl, thấy {labels}")
    n_hard = int(indexed.loc[splits["urfd"]["test"], "adl_hard_neg"].sum())
    n_adl = int((indexed.loc[splits["le2i"]["test"], "label"] == "adl").sum())
    if n_hard < MIN_URFD_TEST_HARD_NEG or n_adl < MIN_LE2I_TEST_ADL:
        raise ValueError(f"test thiếu mẫu: URFD hard-neg={n_hard}, Le2i ADL={n_adl}")


def make_splits(meta: pd.DataFrame, seed: int = 42, ratio: tuple[float, float, float] = DEFAULT_RATIO) -> dict:
    """Tạo nội dung splits.json cho cả 2 bộ và kiểm tra các ràng buộc."""
    splits = {dataset: split_dataset(meta[meta["dataset"] == dataset], seed, ratio) for dataset in ("urfd", "le2i")}
    validate_splits(splits, meta)
    return {**splits, "seed": seed, "ratio": list(ratio)}


def main() -> None:
    """Đọc metadata, chia tập, ghi splits.json và in số video từng tập."""
    parser = argparse.ArgumentParser(description="Chia tập theo video, phân tầng, chống rò rỉ.")
    parser.add_argument("--meta", required=True, help="đường dẫn metadata.csv")
    parser.add_argument("--out", required=True, help="đường dẫn splits.json sẽ ghi")
    parser.add_argument("--seed", type=int, default=42, help="seed chia tập")
    args = parser.parse_args()
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES, keep_default_na=True)
    splits = make_splits(meta, args.seed)
    Path(args.out).write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    indexed = meta.set_index("video_id")
    for dataset in ("urfd", "le2i"):
        for part in PARTS:
            labels = indexed.loc[splits[dataset][part], "label"].value_counts().to_dict()
            hard = int(indexed.loc[splits[dataset][part], "adl_hard_neg"].sum())
            print(f"{dataset}/{part}: {len(splits[dataset][part])} video {labels} | adl_hard_neg={hard}")
    n_left = int((meta["has_event_label"] == 0).sum())
    print(f"TÓM TẮT | đã chia: {len(meta) - n_left} video | bỏ qua: {n_left} (has_event_label = 0) | Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
