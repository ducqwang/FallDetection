"""Unit test bộ đánh giá bằng dữ liệu giả lập (Việc 6, mục 8).

Đầu vào: không có. Kiểm tra: đúng giờ, sớm/muộn vượt T, 2 cảnh báo 1 sự kiện, ADL, không cảnh báo, merge, FAR, frame.
Ví dụ: python -m pytest tests/test_event_metrics.py -q
"""
import numpy as np
import pandas as pd
import pytest
from conftest import meta_frame, meta_row

from src.eval.event_metrics import check_predictions, evaluate, find_alarms, match_alarms, max_score_per_frame

FPS = 30.0
EVENT = (100, 130)  # [3.33 s, 4.33 s]; T = 1 s cho cửa sổ frame [70, 160]


def predictions(video_id, frames, score=0.9, track_id=1):
    return pd.DataFrame({"video_id": video_id, "frame_idx": list(frames), "track_id": track_id, "fall_score": score})


def test_alarm_on_time_is_tp():
    result = match_alarms([110], [EVENT], FPS, 1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert result["latencies_s"] == pytest.approx([10 / FPS])


@pytest.mark.parametrize("frame", [69, 161])
def test_alarm_beyond_tolerance_is_fp_and_fn(frame):
    result = match_alarms([frame], [EVENT], FPS, 1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


@pytest.mark.parametrize("frame", [70, 160])
def test_alarm_on_window_edge_is_tp(frame):
    assert match_alarms([frame], [EVENT], FPS, 1.0)["tp"] == 1


def test_early_alarm_keeps_negative_latency():
    assert match_alarms([85], [EVENT], FPS, 1.0)["latencies_s"][0] == pytest.approx(-0.5)


def test_two_alarms_one_event_earliest_counts_other_ignored():
    result = match_alarms([140, 90], [EVENT], FPS, 1.0)
    assert (result["tp"], result["fp"], result["fn"], result["ignored"]) == (1, 0, 0, 1)
    assert result["latencies_s"] == pytest.approx([-10 / FPS])


def test_alarm_in_adl_is_fp():
    result = match_alarms([50], [], FPS, 1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 0)


def test_no_alarm_is_fn():
    result = match_alarms([], [EVENT], FPS, 1.0)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_tolerance_parameter_changes_outcome():
    assert match_alarms([40], [EVENT], FPS, 2.0)["tp"] == 1
    assert match_alarms([40], [EVENT], FPS, 0.5)["tp"] == 0


def test_tolerance_uses_video_fps():
    assert match_alarms([70], [(95, 120)], 25.0, 1.0)["tp"] == 1  # 95 - 25 = 70
    assert match_alarms([69], [(95, 120)], 25.0, 1.0)["tp"] == 0


def test_contiguous_frames_form_one_alarm():
    scores = np.zeros(100)
    scores[10:20] = 0.9
    assert find_alarms(scores, FPS, 0.5, 1.0) == [(10, 19)]


def test_merge_gap_strictly_below():
    scores = np.zeros(200)
    scores[[10, 39]] = 0.9   # cách 29 frame < 1 s: gộp
    scores[[100, 130]] = 0.9  # cách đúng 30 frame = 1 s: không gộp
    assert find_alarms(scores, FPS, 0.5, 1.0) == [(10, 39), (100, 100), (130, 130)]


def test_threshold_is_inclusive():
    assert find_alarms(np.array([0.0, 0.5, 0.0]), FPS, 0.5, 1.0) == [(1, 1)]


def test_max_score_over_all_tracks():
    table = pd.concat([predictions("v", [1, 2], 0.2, 1), predictions("v", [2, 3], 0.8, 2)])
    assert max_score_per_frame(table, 5).tolist() == [0.0, 0.2, 0.8, 0.8, 0.0]


def test_evaluate_aggregates_and_counts_far_on_all_hours():
    meta = meta_frame([meta_row("fall", "fall", 3600, start=100, end=130),
                       meta_row("adl", "adl", 3600, hard=1)])
    table = pd.concat([predictions("fall", range(110, 115)), predictions("adl", range(10, 15))])
    result = evaluate(table, meta, thr=0.5)
    event = result["event"]
    assert (event["tp"], event["fp"], event["fn"]) == (1, 1, 0)
    assert event["precision"] == 0.5 and event["recall"] == 1.0
    assert event["total_hours"] == pytest.approx(240 / 3600)
    assert event["far_per_hour"] == pytest.approx(15.0)
    assert event["far_nonfall_per_hour"] > event["far_per_hour"]
    assert event["latency_mean_s"] == pytest.approx(event["latency_median_s"])
    hard = result["hard_negative"]
    assert (hard["n_videos"], hard["n_videos_with_fp"], hard["fp"]) == (1, 1, 1)


def test_video_without_predictions_is_fn():
    meta = meta_frame([meta_row("fall", "fall", 300, start=100, end=130), meta_row("adl", "adl", 300)])
    result = evaluate(predictions("adl", [1]).iloc[0:0], meta, thr=0.5)
    assert result["event"]["fn"] == 1 and result["event"]["fp"] == 0
    assert result["event"]["latency_mean_s"] is None


def test_odd_video_is_rejected():
    meta = meta_frame([meta_row("odd", None, has_label=0)])
    with pytest.raises(ValueError):
        evaluate(predictions("odd", [1]), meta)


def test_frame_metrics_skip_unlabeled_adl_frames():
    meta = meta_frame([meta_row("fall", "fall", 100, start=10, end=19), meta_row("adl", "adl", 100)])
    table = pd.concat([predictions("fall", range(10, 20)), predictions("adl", range(0, 20))])
    labels = pd.DataFrame({"video_id": "adl", "frame_idx": range(50, 100), "label": -1})
    with_labels = evaluate(table, meta, thr=0.5, frame_labels=labels)["frame"]
    without = evaluate(table, meta, thr=0.5)["frame"]
    assert with_labels["n_frames"] == 150 and without["n_frames"] == 200
    assert with_labels["n_fall_frames"] == 10  # khoảng ngã tính cả hai đầu
    assert with_labels["accuracy"] == 1.0 and with_labels["roc_auc"] == 1.0 and with_labels["pr_auc"] == 1.0
    assert without["accuracy"] == pytest.approx(180 / 200)


def test_frame_metrics_one_class_gives_none():
    result = evaluate(predictions("adl", [1]), meta_frame([meta_row("adl", "adl", 50)]))["frame"]
    assert result["roc_auc"] is None and result["pr_auc"] is None


@pytest.mark.parametrize("change", [{"fall_score": 1.5}, {"frame_idx": 300}, {"video_id": "khac"}])
def test_check_predictions_rejects_bad_input(change):
    meta = meta_frame([meta_row("adl", "adl", 300)])
    table = predictions("adl", [1]).assign(**change)
    with pytest.raises(ValueError):
        check_predictions(table, meta)
