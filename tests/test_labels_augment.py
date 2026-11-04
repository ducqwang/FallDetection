"""Test labels.py và augment.py (Việc 3).

Đầu vào: dữ liệu giả. Kiểm tra: min_overlap, ADL, chọn track chính, tham số seed, số frame khi đổi tốc độ.
Ví dụ: python -m pytest tests/test_labels_augment.py -q
"""
import numpy as np
import pandas as pd
import pytest

from src.data.augment import (DEFAULT_AUG_CONFIG, augment_clip, blur, brightness_contrast, change_speed, crop_resize,
                              horizontal_flip, save_augment_preview)
from src.data.labels import bbox_iou, select_main_track, window_label

FALL = {"video_id": "v", "label": "fall", "fall_start_frame": 100, "fall_end_frame": 130}
ADL = {"video_id": "a", "label": "adl", "fall_start_frame": None, "fall_end_frame": None}


def test_window_label_overlap_boundary():
    assert window_label(FALL, 85, 114) == 1   # chồng đúng 15/30 frame
    assert window_label(FALL, 84, 113) == 0   # chồng 14/30 frame
    assert window_label(FALL, 0, 50) == 0


def test_window_label_adl_always_zero():
    assert window_label(ADL, 0, 1000) == 0


def test_window_label_rejects_unknown_class():
    with pytest.raises(ValueError):
        window_label({"video_id": "odd", "label": None}, 0, 10)


def tracks(track_id, frames, box):
    return pd.DataFrame({"track_id": track_id, "frame_idx": list(frames), "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})


LE2I = {"video_id": "v", "dataset": "le2i", "width": 320, "height": 240}
URFD = {"video_id": "v", "dataset": "urfd", "width": 640, "height": 480}


def test_iou_values():
    boxes = np.array([[0.0, 0, 10, 10]])
    assert bbox_iou(boxes, boxes)[0] == 1.0
    assert bbox_iou(boxes, np.array([[20.0, 20, 30, 30]]))[0] == 0.0
    assert bbox_iou(boxes, np.array([[5.0, 0, 15, 10]]))[0] == pytest.approx(1 / 3)


def test_urfd_main_track_is_longest():
    table = pd.concat([tracks(1, range(5), (0, 0, 9, 9)), tracks(2, range(20), (0, 0, 9, 9))])
    assert select_main_track(table, URFD) == 2


def test_le2i_main_track_uses_iou_not_length():
    gt = tracks(0, range(10), (10, 10, 50, 100)).drop(columns="track_id")
    table = pd.concat([tracks(1, range(40), (200, 10, 240, 100)), tracks(2, range(10), (10, 10, 50, 100))])
    assert select_main_track(table, LE2I, gt) == 2


def test_le2i_short_perfect_track_loses_to_full_coverage():
    gt = tracks(0, range(100), (10, 10, 50, 100)).drop(columns="track_id")
    table = pd.concat([tracks(1, [0], (10, 10, 50, 100)), tracks(2, range(100), (12, 10, 50, 100))])
    assert select_main_track(table, LE2I, gt) == 2


def test_le2i_requires_ground_truth_and_empty_gives_none():
    with pytest.raises(ValueError):
        select_main_track(tracks(1, [0], (0, 0, 9, 9)), LE2I)
    assert select_main_track(tracks(1, [], (0, 0, 9, 9)), URFD) is None


def make_clip(n_frames=20, height=48, width=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n_frames, height, width, 3), dtype=np.uint8)


def test_each_transform_keeps_frame_size():
    clip = make_clip()
    assert (horizontal_flip(clip)[:, :, ::-1] == clip).all()
    for result in (brightness_contrast(clip, 0.2, 0.2), blur(clip, 3), crop_resize(clip, 0.9, 0.5, 0.5)):
        assert result.shape == clip.shape and result.dtype == np.uint8


def test_change_speed_drops_or_repeats():
    clip = make_clip(20)
    assert len(change_speed(clip, 2.0)) == 10
    assert len(change_speed(clip, 0.5)) == 40


def test_augment_seed_is_reproducible():
    clip = make_clip()
    first, applied = augment_clip(clip, seed=42)
    second, _ = augment_clip(clip, seed=42)
    assert (first == second).all() and isinstance(applied, list)


def test_augment_can_be_switched_off_by_config():
    config = {key: {**value, "p": 0.0} for key, value in DEFAULT_AUG_CONFIG.items()}
    result, applied = augment_clip(make_clip(), config, seed=1)
    assert applied == [] and (result == make_clip()).all()


def test_preview_is_written(tmp_path):
    assert save_augment_preview(make_clip(), tmp_path / "a.png").stat().st_size > 0
