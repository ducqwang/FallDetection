"""Test chia tập và chống rò rỉ (Việc 2).

Đầu vào: metadata giả đúng phân bố thật. Kiểm tra: không trùng tập, đủ lớp, negative khó, ADL Le2i, seed, camera.
Ví dụ: python -m pytest tests/test_split.py -q
"""
import pandas as pd
import pytest
from conftest import meta_row

from src.data.common import sequence_id
from src.data.split import PARTS, make_splits, validate_splits


def test_no_video_in_two_parts(full_meta):
    splits = make_splits(full_meta)
    for dataset in ("urfd", "le2i"):
        ids = [v for part in PARTS for v in splits[dataset][part]]
        assert len(ids) == len(set(ids))


def test_every_part_has_both_classes(full_meta):
    splits = make_splits(full_meta)
    indexed = full_meta.set_index("video_id")
    for dataset in ("urfd", "le2i"):
        for part in PARTS:
            assert set(indexed.loc[splits[dataset][part], "label"]) == {"fall", "adl"}


def test_test_part_has_hard_negative_and_le2i_adl(full_meta):
    splits = make_splits(full_meta)
    indexed = full_meta.set_index("video_id")
    assert indexed.loc[splits["urfd"]["test"], "adl_hard_neg"].sum() >= 1
    assert (indexed.loc[splits["le2i"]["test"], "label"] == "adl").sum() >= 3


def test_same_seed_same_result(full_meta):
    assert make_splits(full_meta, seed=42) == make_splits(full_meta.sample(frac=1, random_state=1), seed=42)


def test_other_seed_changes_result(full_meta):
    assert make_splits(full_meta, seed=42)["urfd"] != make_splits(full_meta, seed=7)["urfd"]


def test_only_labeled_videos_and_counts(full_meta):
    splits = make_splits(full_meta)
    assert sum(len(splits["urfd"][part]) for part in PARTS) == 70
    le2i_ids = [v for part in PARTS for v in splits["le2i"][part]]
    assert len(le2i_ids) == 127 and not any(v.startswith("le2i_coffee_room02_09") for v in le2i_ids)
    assert splits["seed"] == 42 and splits["ratio"] == [0.7, 0.15, 0.15]


def test_ratio_is_close_to_70_15_15(full_meta):
    splits = make_splits(full_meta)
    assert abs(len(splits["le2i"]["train"]) / 127 - 0.70) < 0.02
    assert abs(len(splits["urfd"]["train"]) / 70 - 0.70) < 0.02


def test_cameras_of_a_sequence_stay_together(full_meta):
    cam1 = [meta_row(f"urfd_fall-{i:02d}_cam1", "fall", start=50, end=80) for i in range(1, 31)]
    table = pd.concat([full_meta, pd.DataFrame(cam1)], ignore_index=True)
    splits = make_splits(table)
    for i in range(1, 31):
        owners = [part for part in PARTS for v in splits["urfd"][part] if sequence_id(v) == f"urfd_fall-{i:02d}"]
        assert len(owners) == 2 and owners[0] == owners[1]


def test_validate_catches_leak(full_meta):
    splits = make_splits(full_meta)
    splits["urfd"]["val"].append(splits["urfd"]["train"][0])
    with pytest.raises(ValueError):
        validate_splits(splits, full_meta)
