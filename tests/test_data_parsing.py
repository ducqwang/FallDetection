"""Test đọc URFD, kiểm tra số liệu và chuyển video theo các tình huống đã kiểm tra thật (mục 2.1, 2.2).

Đầu vào: file nhỏ tự tạo trong tmp_path. Kiểm tra: ghép theo số frame, tìm CSV theo đuôi, báo video lỗi, báo mp4 lệch frame.
Ví dụ: python -m pytest tests/test_data_parsing.py -q
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data import checks, prepare_data, urfd
from src.data.common import DataCheckError, le2i_video_id, sequence_id, urfd_video_id
from src.data.metadata import urfd_row

EXTRA = [0.0] * 8


def csv_rows(seq, frames, labels):
    return [[seq, frame, label, *EXTRA] for frame, label in zip(frames, labels)]


def write_csvs(folder, falls, adls):
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / "urfall-cam0-falls.csv", folder / "urfall-cam0-adls.csv"
    pd.DataFrame(falls).to_csv(paths[0], header=False, index=False)
    pd.DataFrame(adls).to_csv(paths[1], header=False, index=False)
    return paths


def sequence(seq, n_images):
    return {"seq": seq, "camera": "cam0", "label": seq.split("-")[0], "frames": [], "numbers": list(range(1, n_images + 1)),
            "width": 640, "height": 480}


def test_csv_found_by_suffix_not_substring(tmp_path):
    falls, adls = write_csvs(tmp_path / "labels", [["fall-01", 1, -1, *EXTRA]], [["adl-01", 1, -1, *EXTRA]])
    assert urfd.find_label_csv(tmp_path, "-falls.csv") == falls
    assert urfd.find_label_csv(tmp_path, "-adls.csv") == adls


def test_swapped_csv_files_are_rejected(tmp_path):
    falls, adls = write_csvs(tmp_path, [["fall-01", 1, -1, *EXTRA]], [["adl-01", 1, -1, *EXTRA]])
    with pytest.raises(DataCheckError):
        urfd.read_labels(adls, falls)


def test_find_urfd_root(tmp_path):
    root = tmp_path / "a" / "b"
    for name in ("adl-01-cam0-rgb", "fall-01-cam0-rgb"):
        (root / name).mkdir(parents=True)
    assert urfd.find_urfd_root(tmp_path) == root


def test_labels_join_by_frame_number_not_row_order():
    # adl: CSV bắt đầu từ frame 6, thiếu frame 7, kết thúc ở frame 8; ảnh có 10 frame
    falls = pd.DataFrame(csv_rows("fall-01", [1, 2, 3], [-1, 0, 1]), columns=urfd.CSV_COLUMNS)
    adls = pd.DataFrame(csv_rows("adl-01", [6, 8], [-1, 1]), columns=urfd.CSV_COLUMNS)
    labels = pd.concat([falls, adls])[urfd.USED_CSV_COLUMNS]
    fall, adl = urfd.match_labels([sequence("fall-01", 3), sequence("adl-01", 10)], labels)
    assert fall["status"] == "OK" and fall["first_label0"] == 2 and fall["first_label1"] == 3
    assert adl["status"] == "CSV_THIEU_FRAME"
    assert (adl["miss_head"], adl["gap_mid"], adl["miss_tail"]) == (5, 1, 2)
    assert (adl["label_first_frame"], adl["label_last_frame"]) == (5, 7)


def test_csv_row_without_image_is_flagged():
    labels = pd.DataFrame(csv_rows("adl-01", [1, 99], [-1, -1]), columns=urfd.CSV_COLUMNS)[urfd.USED_CSV_COLUMNS]
    assert urfd.match_labels([sequence("adl-01", 5)], labels)[0]["status"] == "LECH_THAT"


def test_frame_label_table_is_zero_based():
    labels = pd.DataFrame({"seq": ["adl-01"], "frame": [6], "label": [1]})
    assert urfd.frame_label_table(labels)["frame_idx"].tolist() == [5]


def test_urfd_row_fall_zero_based_and_adl_label_does_not_create_event():
    fall = {**sequence("fall-01", 100), "n_images": 100, "n_csv_rows": 100, "first_label0": 83, "first_label1": 113,
            "n_label0": 30, "n_label1": 5, "label_first_frame": 0, "label_last_frame": 99}
    adl = {**sequence("adl-01", 100), "n_images": 100, "n_csv_rows": 50, "first_label0": 60, "first_label1": 70,
           "n_label0": 10, "n_label1": 30, "label_first_frame": 50, "label_last_frame": 99}
    fall_row, adl_row = urfd_row(fall, 30), urfd_row(adl, 30)
    assert (fall_row["fall_start_frame"], fall_row["fall_end_frame"]) == (82, 112)
    assert fall_row["video_id"] == "urfd_fall-01_cam0" and fall_row["adl_hard_neg"] == 0
    assert fall_row["label_source"] == adl_row["label_source"] == "urfd_csv"
    assert adl_row["fall_start_frame"] is None and adl_row["fall_end_frame"] is None
    assert adl_row["adl_hard_neg"] == 1 and adl_row["label_coverage"] == 0.5


def test_video_ids_and_sequence_id():
    assert urfd_video_id("adl-01", "cam0") == "urfd_adl-01_cam0"
    assert le2i_video_id("coffee_room", "01", 12) == "le2i_coffee_room01_012"
    assert sequence_id("urfd_fall-05_cam1") == sequence_id("urfd_fall-05_cam0") == "urfd_fall-05"
    assert sequence_id("le2i_home02_031") == "le2i_home02_031"


def test_check_reports_mismatch_instead_of_passing():
    issues = checks._differs("x", 1, 2)
    assert issues == ["x: đo được 1, mong đợi 2"] and checks._differs("x", 1, 1) == []


def test_repo_config_has_expected_numbers():
    import yaml
    config = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
    assert len(config["expect"]["urfd"]["adl_csv_thieu_frame"]) == 18
    parts = config["expect"]["le2i"]["per_part"].values()
    assert [sum(v[index] for v in parts) for index in range(3)] == [130, 99, 31]
    header_mid = config["expect"]["le2i"]["header_mid"]
    assert header_mid["coffee_room|01|26"] == {"start": 197, "end": 227, "lines": [219, 220], "last_frame": 278}
    assert header_mid["coffee_room|02|50"] == {"start": 1816, "end": 1852, "lines": [1894, 1895], "last_frame": 1954}
    assert header_mid["coffee_room|02|52"] == {"start": 87, "end": 113, "lines": [156, 157], "last_frame": 215}


def test_read_video_collects_bad_annotation_instead_of_stopping(monkeypatch, tmp_path):
    bad = tmp_path / "video (1).txt"
    bad.write_text("\n".join(["48", "80", "197", "1,1,1,1,5,5"]))
    monkeypatch.setattr(prepare_data.media, "probe_video", lambda path: {"fps": 25.0, "width": 320, "height": 240})
    monkeypatch.setattr(prepare_data.media, "count_frames", lambda path: 30)
    video = {"key": "home|01|1", "scene": "home", "part": "01", "number": 1, "path": tmp_path / "a.avi", "annotation": bad}
    assert "không đủ 6 cột" in prepare_data._read_le2i_video(video)["read_error"]


def fake_measure(monkeypatch, n_frames):
    monkeypatch.setattr(prepare_data.media, "probe_video", lambda path: {"fps": 25.0, "width": 320, "height": 240})
    monkeypatch.setattr(prepare_data.media, "count_frames", lambda path: n_frames)
    monkeypatch.setattr(prepare_data.media, "avi_to_mp4", lambda source, out: None)


def mismatch_meta():
    return pd.DataFrame([{"video_id": "le2i_home01_001", "path": "videos/a.mp4", "fps": 25.0, "n_frames": 30,
                          "width": 320, "height": 240}])


def le2i_source():
    return [{"row": {"video_id": "le2i_home01_001"}, "path": Path("video (1).avi")}]


def test_le2i_frame_mismatch_names_video_counts_and_hint(monkeypatch, tmp_path, capsys):
    fake_measure(monkeypatch, 31)
    with pytest.raises(DataCheckError) as error:
        prepare_data.convert_videos(mismatch_meta(), [], le2i_source(), tmp_path, 1, False)
    printed = capsys.readouterr().out
    for text in ("le2i_home01_001", "số frame nguồn 30", "số frame mp4 31", "-fps_mode passthrough", "-vsync 0"):
        assert text in str(error.value) and text in printed


def test_urfd_frame_mismatch_has_no_le2i_hint(monkeypatch, tmp_path):
    fake_measure(monkeypatch, 31)
    monkeypatch.setattr(prepare_data.media, "png_sequence_to_mp4", lambda frames, out, fps: None)
    meta = mismatch_meta().assign(video_id="urfd_fall-01_cam0")
    sequences = [{"seq": "fall-01", "camera": "cam0", "frames": []}]
    with pytest.raises(DataCheckError) as error:
        prepare_data.convert_videos(meta, sequences, [], tmp_path, 1, False)
    assert "số frame mp4 31" in str(error.value) and "fps_mode" not in str(error.value)


def test_run_log_uses_counted_mismatch(monkeypatch, tmp_path):
    fake_measure(monkeypatch, 30)
    meta = mismatch_meta().assign(dataset="le2i", label="fall", has_event_label=1)
    n_ok, n_mismatch = prepare_data.convert_videos(meta, [], le2i_source(), tmp_path, 1, False)
    assert (n_ok, n_mismatch) == (1, 0)
    assert "mp4 frame mismatch: 7" in prepare_data.run_log(meta, 7)
