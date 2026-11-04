"""Test đọc URFD và Le2i theo các tình huống đã kiểm tra thật (mục 2.1, 2.2).

Đầu vào: file nhỏ tự tạo trong tmp_path. Kiểm tra: ghép theo số frame, tìm CSV theo đuôi, annotation, clip bbox.
Ví dụ: python -m pytest tests/test_data_parsing.py -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import checks, le2i, prepare_data, urfd
from src.data.common import DataCheckError, le2i_video_id, sequence_id, urfd_video_id
from src.data.metadata import le2i_row, metadata_frame, urfd_row

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
    assert adl_row["fall_start_frame"] is None and adl_row["fall_end_frame"] is None
    assert adl_row["adl_hard_neg"] == 1 and adl_row["label_coverage"] == 0.5


def le2i_tree(root):
    for folder, annot, number, text in (("Home_01", "Annotation_files", 1, "1\n2\n1,1,1,1,5,5\n"),
                                        ("Coffee_room_02", "Annotations_files", 1, "0\n0\n1,1,1,1,5,5\n")):
        videos = root / folder / folder / "Videos"
        notes = root / folder / folder / annot
        videos.mkdir(parents=True)
        notes.mkdir(parents=True)
        (videos / f"video ({number}).avi").touch()
        (notes / f"video ({number}).txt").write_text(text)
    office = root / "Office" / "Office"
    office.mkdir(parents=True)
    (office / "video (1).avi").touch()


def test_le2i_same_number_different_scene_and_annotation_folder_names(tmp_path):
    le2i_tree(tmp_path)
    assert le2i.find_le2i_root(tmp_path.parent) == tmp_path
    videos = {v["key"]: v for v in le2i.scan_videos(tmp_path)}
    assert set(videos) == {"home|01|1", "coffee_room|02|1", "office||1"}
    assert videos["home|01|1"]["annotation"].parent.name == "Annotation_files"
    assert videos["coffee_room|02|1"]["annotation"].parent.name == "Annotations_files"
    assert videos["office||1"]["annotation"] is None


def annotation(tmp_path, text):
    path = tmp_path / "video (1).txt"
    path.write_text(text)
    return le2i.parse_annotation(path)


def test_parse_fall_header_and_drop_second_column(tmp_path):
    parsed = annotation(tmp_path, "48\n80\n1,1,272,48,312,207\n")
    assert (parsed["kind"], parsed["start"], parsed["end"]) == ("fall", 48, 80)
    assert parsed["boxes"].tolist() == [[1, 272, 48, 312, 207]]


def test_parse_adl_and_odd(tmp_path):
    assert annotation(tmp_path, "0\n0\n1,1,1,1,5,5\n")["kind"] == "adl"
    odd = annotation(tmp_path, "1,1,72,58,132,170\n2,1,72,58,132,170\n")
    assert odd["kind"] == "odd" and odd["start"] is None and len(odd["boxes"]) == 2


def test_parse_rejects_reversed_interval(tmp_path):
    with pytest.raises(DataCheckError):
        annotation(tmp_path, "50\n10\n1,1,1,1,5,5\n")


def test_clean_boxes_zero_rows_clip_and_zero_based():
    boxes = np.array([[1, 0, 0, 0, 0], [2, -5, 10, 330, 250], [3, 325, 10, 340, 50]], dtype=float)
    table, stats = le2i.clean_boxes("v", boxes, 320, 240, 10)
    assert stats == {"n_rows": 3, "n_zero": 1, "n_degenerate": 1}
    assert table.iloc[0][["frame_idx", "x1", "y1", "x2", "y2"]].tolist() == [1, 0, 10, 320, 240]


def test_clean_boxes_rejects_frame_outside_video():
    with pytest.raises(DataCheckError):
        le2i.clean_boxes("v", np.array([[11, 1, 1, 5, 5]], dtype=float), 320, 240, 10)


def test_le2i_row_fall_and_odd():
    base = {"scene": "home", "part": "02", "number": 31, "fps": 24.0, "n_frames": 240, "width": 320, "height": 180}
    fall = le2i_row({**base, "kind": "fall", "start": 48, "end": 80})
    odd = le2i_row({**base, "kind": "odd", "start": None, "end": None})
    assert fall["video_id"] == "le2i_home02_031" and fall["fall_start_frame"] == 47 and fall["fall_end_frame"] == 79
    assert fall["duration_s"] == 10.0 and fall["has_event_label"] == 1
    assert odd["has_event_label"] == 0 and odd["label"] == "" and odd["fall_start_frame"] is None


def test_metadata_frame_keeps_integers_and_rejects_duplicates():
    base = {"scene": "home", "part": "02", "number": 1, "fps": 24.0, "n_frames": 240, "width": 320, "height": 180}
    row = le2i_row({**base, "kind": "fall", "start": 48, "end": 80})
    assert str(metadata_frame([row])["fall_start_frame"].dtype) == "Int64"
    with pytest.raises(ValueError):
        metadata_frame([row, row])


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
    assert sum(v[0] for v in config["expect"]["le2i"]["per_part"].values()) == 130


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
