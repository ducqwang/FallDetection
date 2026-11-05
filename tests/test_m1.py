"""Test M1 trên track giả lập có quỹ đạo té ngã rõ ràng.

Đầu vào: track tự sinh. Kiểm tra: phân biệt ngã, không phụ thuộc độ phân giải và fps, bất động không bắt buộc, chỉ tune trên val.
Ví dụ: python -m pytest tests/test_m1.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import meta_frame, meta_row

from src.data.common import PREDICTION_COLUMNS
from src.eval.event_metrics import evaluate
from src.models import m1_rule
from src.models.m1_features import (contiguous_track, run_length_seconds, seconds_to_frames, split_at_long_gaps,
                                    track_signals)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import tune_m1  # noqa: E402

CONFIG = m1_rule.load_config("configs/m1.yaml")
THR = CONFIG["alarm"]["thr"]


def fall_track(fps=30.0, scale=1.0, stand_s=2.0, fall_s=0.7, lie_s=2.0, video_id="v", track_id=1):
    """Đứng rồi ngã: bbox từ 60×170 thành 170×60 trong fall_s giây, tâm đi xuống."""
    n_stand, n_fall, n_lie = (round(s * fps) for s in (stand_s, fall_s, lie_s))
    progress = np.r_[np.zeros(n_stand), np.linspace(0, 1, n_fall), np.ones(n_lie)]
    width, height = 60 + 110 * progress, 170 - 110 * progress
    center_y = 130 + 100 * progress
    return pd.DataFrame({"video_id": video_id, "frame_idx": np.arange(len(progress)), "track_id": track_id,
                         "x1": (200 - width / 2) * scale, "y1": (center_y - height / 2) * scale,
                         "x2": (200 + width / 2) * scale, "y2": (center_y + height / 2) * scale, "det_conf": 0.9})


def score_of(track, fps=30.0):
    meta = pd.DataFrame({"video_id": [track["video_id"].iloc[0]], "fps": [fps]})
    return m1_rule.predict(track, meta, CONFIG)["fall_score"].to_numpy()


def standing_track(n=300):
    track = fall_track(stand_s=n / 30, fall_s=0, lie_s=0)
    return track


def test_fall_passes_threshold_and_standing_does_not():
    assert score_of(fall_track()).max() >= THR
    assert score_of(standing_track()).max() < THR


def test_score_stays_in_unit_interval():
    scores = score_of(fall_track())
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_fall_score_is_causal():
    track = fall_track()
    full = score_of(track)
    cut = score_of(track[track["frame_idx"] < 75])
    assert full[:75] == pytest.approx(cut)


def test_alarm_arrives_inside_the_fall_window():
    scores = score_of(fall_track())
    first = int(np.argmax(scores >= THR))
    assert 60 <= first <= 60 + 21 + 30


def test_short_stay_on_floor_still_alarms():
    """fall-21 chỉ còn 5 frame sau khi nằm: bất động không được là điều kiện bắt buộc."""
    assert score_of(fall_track(lie_s=5 / 30)).max() >= THR


def test_slow_lying_down_scores_lower_than_fall():
    assert score_of(fall_track(fall_s=4.0)).max() < score_of(fall_track()).max()


def test_lying_person_from_start_does_not_alarm():
    track = fall_track(stand_s=0, fall_s=0, lie_s=5)
    assert score_of(track).max() < THR


def test_resolution_invariant():
    assert score_of(fall_track(scale=1.0)) == pytest.approx(score_of(fall_track(scale=0.5)), abs=1e-6)


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
def test_works_at_each_dataset_fps(fps):
    assert score_of(fall_track(fps=fps), fps).max() >= THR


def test_seconds_are_converted_with_video_fps():
    assert seconds_to_frames(0.5, 30) == 15 and seconds_to_frames(0.5, 24) == 12
    assert seconds_to_frames(0.001, 25) == 1


def test_run_length_counts_seconds_and_resets():
    active = pd.Series([True, True, False, True, True, True])
    assert run_length_seconds(active, 2.0).tolist() == [0.5, 1.0, 0.0, 0.5, 1.0, 1.5]


def test_still_score_needs_time_after_posture_change():
    rules, weights = CONFIG["rules"], {"speed": 0.0, "accel": 0.0, "posture": 0.0, "still": 1.0}
    signals = pd.DataFrame({"aspect": [0.5] * 30 + [2.0] * 30, "vy": 0.0, "ay": 0.0, "speed": 0.0})
    scores = m1_rule.score_signals(signals, 30.0, rules, weights)
    assert scores.iloc[29] == 0.0 and scores.iloc[37] == pytest.approx(8 / 15, abs=1e-9) and scores.iloc[-1] == 1.0


def test_aspect_is_width_over_height_and_thresholds_go_up():
    signals = track_signals(fall_track(), 30.0, 0.2, 1.5)
    assert signals["aspect"].iloc[30] < 1.0 < signals["aspect"].iloc[-1]
    assert CONFIG["rules"]["ar_upright"] < CONFIG["rules"]["ar_lying"]


def test_posture_score_high_when_person_has_just_lain_down():
    """Đứng 2 s, nằm xuống trong 1 s, nằm yên 2 s."""
    signals = track_signals(fall_track(stand_s=2, fall_s=1, lie_s=2), 30.0, 0.2, 1.5)
    just_lying = int(np.argmax(signals["aspect"].to_numpy() >= CONFIG["rules"]["ar_lying"]))
    scores = m1_rule.posture_score(signals, 30.0, CONFIG["rules"])
    assert 60 <= just_lying <= 90 and scores.iloc[just_lying] > 0.5
    assert scores.iloc[:60].max() == 0.0 and scores.iloc[-1] == 0.0


def test_ramp_rejects_low_not_below_high():
    with pytest.raises(ValueError):
        m1_rule.ramp(pd.Series([1.0]), 1.0, 1.0)


def test_every_grid_combination_keeps_low_below_high():
    grid, rules = CONFIG["grid"], CONFIG["rules"]
    assert max(grid["ar_upright"]) < min(grid["ar_lying"])
    assert rules["vy_min"] < min(grid["vy_full"])


def moving_track(n_frames=120, px_per_frame=1.0):
    """Người đứng đi xuống đều, bbox không đổi."""
    frames = np.arange(n_frames)
    return pd.DataFrame({"video_id": "v", "frame_idx": frames, "track_id": 1, "x1": 100.0, "x2": 160.0,
                         "y1": 40.0 + frames * px_per_frame, "y2": 160.0 + frames * px_per_frame, "det_conf": 0.9})


def test_ten_frame_gap_adds_no_fake_velocity_or_acceleration():
    track = moving_track()
    gapped = track[~track["frame_idx"].between(50, 59)]
    full_signals = track_signals(track, 30.0, 0.2, 1.5)
    gap_signals = track_signals(gapped, 30.0, 0.2, 1.5)
    assert len(gap_signals) == len(full_signals)
    for column in ("vy", "ay"):
        assert gap_signals[column].to_numpy() == pytest.approx(full_signals[column].to_numpy(), abs=1e-9)


def test_short_gap_stays_in_one_segment_long_gap_splits():
    track = moving_track()
    assert len(split_at_long_gaps(track[~track["frame_idx"].between(50, 59)], 30.0, 0.5)) == 1
    assert len(split_at_long_gaps(track[~track["frame_idx"].between(50, 64)], 30.0, 0.5)) == 1  # đúng 0.5 s
    assert len(split_at_long_gaps(track[~track["frame_idx"].between(50, 65)], 30.0, 0.5)) == 2


def test_long_gap_is_not_interpolated_and_resets_state():
    """Mất track 3 s rồi thấy người đã nằm: không có frame bịa, không được tính là vừa ngã."""
    standing, lying = moving_track(60, 0.0), moving_track(60, 0.0)
    lying = lying.assign(frame_idx=lying["frame_idx"] + 150, x2=220.0, y1=120.0, y2=170.0)
    meta = pd.DataFrame({"video_id": ["v"], "fps": [30.0]})
    table = m1_rule.predict(pd.concat([standing, lying]), meta, CONFIG)
    assert not table["frame_idx"].between(60, 149).any() and len(table) == 120
    assert table["fall_score"].max() < THR


def test_lost_frames_are_interpolated():
    track = fall_track().drop(index=[5, 6, 7])
    filled = contiguous_track(track)
    assert list(filled.index) == list(range(filled.index.max() + 1)) and not filled.isna().any().any()


def test_duplicate_frame_is_rejected():
    track = fall_track().iloc[[0, 0]]
    with pytest.raises(ValueError):
        contiguous_track(track)


def test_height_normalization_keeps_lying_noise_small():
    track = fall_track(stand_s=2, fall_s=0.7, lie_s=3)
    signals = track_signals(track, 30.0, 0.2, 1.5)
    assert signals["vy"].iloc[:50].abs().max() < 1e-6 and signals["vy"].max() > 0.3


def test_predictions_have_schema_and_short_track_scores_zero():
    meta = pd.DataFrame({"video_id": ["v"], "fps": [30.0]})
    table = m1_rule.predict(fall_track().head(5), meta, CONFIG)
    assert list(table.columns) == PREDICTION_COLUMNS and (table["fall_score"] == 0).all()


def test_gt_boxes_become_one_track():
    gt = fall_track().drop(columns=["track_id", "det_conf"])
    tracks = m1_rule.gt_boxes_as_tracks(gt)
    assert tracks["track_id"].nunique() == 1 and (tracks["det_conf"] == 1.0).all()


def test_weights_must_sum_to_one(tmp_path):
    config = yaml.safe_load(Path("configs/m1.yaml").read_text(encoding="utf-8"))
    config["weights"]["speed"] = 0.9
    path = tmp_path / "m1.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        m1_rule.load_config(path)


def test_unknown_video_is_rejected():
    with pytest.raises(ValueError):
        m1_rule.predict(fall_track(video_id="khac"), pd.DataFrame({"video_id": ["v"], "fps": [30.0]}), CONFIG)


def test_tuning_picks_best_row_and_ties_by_fewer_fp():
    table = pd.DataFrame({"f1": [0.5, 0.8, 0.8], "fp": [0, 3, 1], "thr": [0.4, 0.5, 0.6]})
    assert tune_m1.best_row(table)["thr"] == 0.6


def test_tuning_search_runs_on_given_videos_only():
    config = {**CONFIG, "grid": {"thr": [0.5, 0.7], "ar_lying": [1.0, 1.2]}}
    meta = meta_frame([meta_row("val_fall", "fall", 400, start=60, end=81), meta_row("val_adl", "adl", 300)])
    tracks = pd.concat([fall_track(video_id="val_fall"), standing_track().assign(video_id="val_adl")])
    table = tune_m1.search(m1_rule.signals_by_track(tracks, meta, config), meta, config)
    assert len(table) == 4 and table["f1"].max() == 1.0


def test_end_to_end_score_to_metrics():
    meta = meta_frame([meta_row("v", "fall", 200 + 21 + 60, start=60, end=81)])
    table = m1_rule.predict(fall_track(), meta, CONFIG)
    assert evaluate(table, meta, thr=THR)["event"]["tp"] == 1
