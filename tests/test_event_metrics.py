"""Unit test bộ đánh giá bằng dữ liệu giả lập (Việc 6, mục 8).

Đầu vào: không có. Kiểm tra: đúng giờ, sớm/muộn vượt T, 2 cảnh báo 1 sự kiện, ADL, không cảnh báo, merge, FAR, frame.
Ví dụ: python -m pytest tests/test_event_metrics.py -q
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import meta_frame, meta_row

from src.eval.event_metrics import (MAIN_FAR_KEY, check_predictions, evaluate, find_alarms, match_alarms,
                                    max_score_per_frame)

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


def test_alarm_starting_before_window_is_fp_and_fn_even_if_it_runs_into_window():
    """Quyết định C4: cảnh báo tính tại frame đầu của đoạn, đoạn 50..110 chạm cửa sổ [70, 160] nhưng bắt đầu trước."""
    meta = meta_frame([meta_row("fall", "fall", 300, start=100, end=130)])
    event = evaluate(predictions("fall", range(50, 111)), meta, thr=0.5)["event"]
    assert (event["tp"], event["fp"], event["fn"]) == (0, 1, 1)


def test_main_far_excludes_tolerance_window_from_hours():
    """Video 10 s, sự kiện 100..130 (3.33..4.33 s), T = 1 s: cửa sổ dài 3 s nên còn 7 s không-ngã."""
    meta = meta_frame([meta_row("fall", "fall", 300, start=100, end=130)])
    event = evaluate(predictions("fall", [10, 200]), meta, thr=0.5)["event"]
    assert event["fp"] == 2
    assert event[MAIN_FAR_KEY] == pytest.approx(2 / (7 / 3600))
    assert event["far_per_hour"] == pytest.approx(2 / (10 / 3600))


def test_summary_table_names_main_far_column(tmp_path):
    meta = meta_frame([meta_row("fall", "fall", 300, start=100, end=130)])
    result = evaluate(predictions("fall", [110]), meta, thr=0.5)
    (tmp_path / "run.json").write_text(json.dumps({"results": [result]}), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/summarize_results.py", "--results-dir", str(tmp_path), "--out",
                    str(tmp_path / "s.csv")], check=True, capture_output=True)
    columns = pd.read_csv(tmp_path / "s.csv").columns.tolist()
    assert "far_nonfall_per_hour (CHÍNH)" in columns and "far_per_hour (phụ)" in columns


def test_video_without_predictions_is_fn():
    meta = meta_frame([meta_row("fall", "fall", 300, start=100, end=130), meta_row("adl", "adl", 300)])
    result = evaluate(predictions("adl", [1]).iloc[0:0], meta, thr=0.5)
    assert result["event"]["fn"] == 1 and result["event"]["fp"] == 0
    assert result["event"]["latency_mean_s"] is None


def test_video_without_event_label_is_rejected():
    meta = meta_frame([meta_row("no_label", None, has_label=0)])
    with pytest.raises(ValueError):
        evaluate(predictions("no_label", [1]), meta)


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


BAD_INPUTS = [({"fall_score": 1.5}, "fall_score"), ({"frame_idx": 300}, "frame_idx"), ({"video_id": "khac"}, "khac")]


@pytest.mark.parametrize("change,message", BAD_INPUTS)
def test_evaluate_raises_value_error_not_index_error(change, message):
    meta = meta_frame([meta_row("adl", "adl", 300)])
    with pytest.raises(ValueError, match=message):
        evaluate(predictions("adl", [1]).assign(**change), meta)


@pytest.mark.parametrize("change,message", BAD_INPUTS)
def test_cli_rejects_bad_predictions_with_clear_message(change, message, tmp_path):
    meta_frame([meta_row("adl", "adl", 300), meta_row("f", "fall", 300, start=10, end=40)]).to_csv(tmp_path / "metadata.csv", index=False)
    predictions("adl", [1]).assign(**change).to_csv(tmp_path / "pred.csv", index=False)
    done = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", str(tmp_path / "pred.csv"), "--meta",
                           str(tmp_path / "metadata.csv"), "--split", "all", "--dataset", "urfd", "--out-dir", str(tmp_path)],
                          capture_output=True, text=True)
    assert done.returncode != 0 and "ValueError" in done.stderr and message in done.stderr
    assert "IndexError" not in done.stderr


def test_predictions_of_other_videos_are_valid_but_not_counted():
    meta = meta_frame([meta_row("val", "adl", 300), meta_row("test", "adl", 300)])
    table = pd.concat([predictions("val", [1]), predictions("test", [1])])
    result = evaluate(table, meta[meta["video_id"] == "val"], thr=0.5, reference_meta=meta)
    assert result["event"]["n_videos"] == 1 and result["event"]["fp"] == 1


@pytest.mark.parametrize("change", [change for change, _ in BAD_INPUTS])
def test_check_predictions_rejects_bad_input(change):
    meta = meta_frame([meta_row("adl", "adl", 300)])
    table = predictions("adl", [1]).assign(**change)
    with pytest.raises(ValueError):
        check_predictions(table, meta)
