"""Chạy prepare_data, split, loader, M1, evaluate, EDA từ đầu đến cuối trên dữ liệu GIẢ. Cần ffmpeg.

Đầu vào: cây thư mục do fake_kaggle sinh. Đầu ra: thư mục tạm của pytest.
Test này chỉ chứng minh luồng xử lý và schema đúng, không nói gì về chất lượng trên dữ liệu thật.
Ví dụ: python -m pytest tests/test_pipeline_fake.py -q
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fake_kaggle
from src.data import loader, media
from src.data.common import FRAME_LABEL_COLUMNS, GT_BBOX_COLUMNS, METADATA_COLUMNS, DataCheckError

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="cần ffmpeg và ffprobe")
PYTHON = sys.executable


def run(*args):
    done = subprocess.run([PYTHON, *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    config = fake_kaggle.build(root)
    log = run("-m", "src.data.prepare_data", "--config", str(config), "--workers", "6")
    assert "urfd: 70 video (30 fall, 40 adl) | le2i: 130 video (99 fall, 31 adl) | mp4 frame mismatch: 0" in log
    assert "video lỗi không đọc được: []" in log and "video thiếu annotation: []" in log
    run("-m", "src.data.split", "--meta", str(root / "out" / "metadata.csv"), "--out", str(root / "out" / "splits.json"))
    return root / "out"


def test_output_tree_and_schema(processed):
    assert (processed / "DATA_CARD.md").exists()
    meta = loader.load_metadata(processed)
    assert list(meta.columns) == METADATA_COLUMNS and len(list((processed / "videos").glob("*.mp4"))) == 200
    assert list(pd.read_csv(processed / "annotations" / "le2i_gt_bbox.csv").columns) == GT_BBOX_COLUMNS
    assert list(pd.read_csv(processed / "annotations" / "urfd_frame_labels.csv").columns) == FRAME_LABEL_COLUMNS


def test_metadata_counts_and_excluded_scenes(processed):
    meta = loader.load_metadata(processed)
    urfd, le2i = meta[meta["dataset"] == "urfd"], meta[meta["dataset"] == "le2i"]
    assert urfd["label"].value_counts().to_dict() == {"adl": 40, "fall": 30}
    assert le2i["label"].value_counts().to_dict() == {"fall": 99, "adl": 31}
    assert int(le2i["has_event_label"].sum()) == 130 and set(le2i["scene"]) == {"coffee_room", "home"}
    assert set(le2i["scene_part"]) == {"01", "02"} and int(urfd["adl_hard_neg"].sum()) == 16
    assert set(le2i["fps"]) == {24.0, 25.0}
    assert le2i["label_source"].value_counts().to_dict() == {"header_top": 127, "header_mid": 3}
    assert set(urfd["label_source"]) == {"urfd_csv"}
    middle = le2i[le2i["label_source"] == "header_mid"]
    assert sorted(middle["video_id"]) == ["le2i_coffee_room01_026", "le2i_coffee_room02_050", "le2i_coffee_room02_052"]
    assert middle["fall_start_frame"].tolist() == [9, 9, 9] and middle["fall_end_frame"].tolist() == [19, 19, 19]


def test_data_card_has_required_sections(processed):
    card = (processed / "DATA_CARD.md").read_text(encoding="utf-8")
    for text in ("dài đúng 30 frame", "Office và Lecture room", "CSV_THIEU_FRAME", "Kwolek", "Charfi", "shahliza27",
                 "Kiểm tra bắt buộc", "URFD Fall khớp hoàn toàn: 30/30", "URFD dòng CSV không có ảnh: 0",
                 "Le2i: 130 video (99 fall, 31 adl)", "Le2i kích thước", "header_mid", "aux_code", "16-17",
                 "danh sách video Le2i negative khó chưa có", "cùng người và cùng cảnh quay", "--split all"):
        assert text in card


def test_loader_reads_exact_frames_and_nan_labels(processed):
    assert loader.read_frames("urfd_fall-16_cam0", 3, 8, processed).shape == (5, 480, 640, 3)
    assert loader.read_frames("le2i_home01_001", root=processed).shape == (30, 180, 320, 3)
    labels = loader.load_frame_labels("urfd_adl-17_cam0", processed)
    assert len(labels) == 230 and labels.isna().sum() > 0 and labels.iloc[0] != labels.iloc[0]
    assert len(loader.get_split("urfd", "test", processed)) > 0


def test_mismatch_stops_before_conversion(tmp_path):
    config = fake_kaggle.build(tmp_path)
    text = config.read_text(encoding="utf-8").replace("n_adl_hard_neg: 16", "n_adl_hard_neg: 15")
    config.write_text(text, encoding="utf-8")
    done = subprocess.run([PYTHON, "-m", "src.data.prepare_data", "--config", str(config)], capture_output=True, text=True)
    assert done.returncode != 0 and DataCheckError.__name__ in done.stderr and "số ADL negative khó" in done.stderr
    assert not (tmp_path / "out" / "videos").exists()


def test_unreadable_video_is_reported_then_stops(tmp_path):
    config = fake_kaggle.build(tmp_path)
    video = next((tmp_path / "input").rglob("Home_01/Videos/video (1).avi"))
    video.write_bytes(b"not a video")
    done = subprocess.run([PYTHON, "-m", "src.data.prepare_data", "--config", str(config)], capture_output=True, text=True)
    assert done.returncode != 0 and "video lỗi home|01|1" in done.stderr
    assert "video lỗi không đọc được: ['home|01|1" in done.stdout
    assert not (tmp_path / "out" / "metadata.csv").exists()


def test_every_bad_annotation_is_reported_at_once(tmp_path):
    config = fake_kaggle.build(tmp_path)
    for folder, number in (("Home_01", 1), ("Home_02", 31)):
        path = next((tmp_path / "input").rglob(f"{folder}/Annotation_files/video ({number}).txt"))
        path.write_text("48\n80\n197\n1,1,1,1,5,5\n", encoding="utf-8")
    done = subprocess.run([PYTHON, "-m", "src.data.prepare_data", "--config", str(config)], capture_output=True, text=True)
    assert done.returncode != 0
    for key in ("home|01|1", "home|02|31"):
        assert f"video lỗi {key}" in done.stderr and key in done.stdout
    assert "không đủ 6 cột: '197'" in done.stderr and not (tmp_path / "out" / "metadata.csv").exists()


def test_m1_gt_mode_evaluate_and_eda(processed, tmp_path):
    meta_path, gt = processed / "metadata.csv", processed / "annotations" / "le2i_gt_bbox.csv"
    predictions = tmp_path / "pred.csv"
    run("-m", "src.models.m1_rule", "--gt-bbox", str(gt), "--meta", str(meta_path), "--out", str(predictions))
    tuned = tmp_path / "tuned.yaml"
    run("scripts/tune_m1.py", "--gt-bbox", str(gt), "--meta", str(meta_path), "--dataset", "le2i", "--out", str(tuned))
    assert tuned.exists() and (tmp_path / "tuned_grid.csv").exists()
    out = run("scripts/evaluate.py", "--pred", str(predictions), "--meta", str(meta_path), "--split", "test",
              "--dataset", "le2i", "--T", "0.5", "1.0", "--out-dir", str(tmp_path))
    assert "TÓM TẮT" in out
    payload = json.loads((tmp_path / "pred_le2i_test.json").read_text(encoding="utf-8"))
    assert [item["params"]["tolerance_s"] for item in payload["results"]] == [0.5, 1.0]
    run("-m", "src.eda.plots", "--meta", str(meta_path), "--gt-bbox", str(gt), "--out-dir", str(tmp_path / "fig"), "--window-s", "0.5")
    assert len(list((tmp_path / "fig").glob("eda_*.png"))) == 7  # 5 hình + 5b theo kích thước + 6 bảng kích thước
    assert (tmp_path / "fig" / "eda_5b_pca_tsne_by_size.png").exists()


def test_png_order_is_numeric_and_frames_are_exact(tmp_path):
    from PIL import Image

    frames = []
    for number in (1, 2, 10, 11):  # sắp theo chữ sẽ thành 1, 10, 11, 2
        path = tmp_path / f"f-{number}.png"
        Image.new("RGB", (64, 48), (number * 20,) * 3).save(path)
        frames.append(path)
    out = tmp_path / "videos" / "x.mp4"
    media.png_sequence_to_mp4(frames, out, 30)
    assert media.count_frames(out) == 4 and media.probe_video(out) == {"fps": 30.0, "width": 64, "height": 48}
    meta = pd.DataFrame([{"video_id": "x", "path": "videos/x.mp4", "n_frames": 4, "width": 64, "height": 48}])
    meta.to_csv(tmp_path / "metadata.csv", index=False)
    brightness = loader.read_frames("x", root=tmp_path).mean(axis=(1, 2, 3))
    assert (np.diff(brightness) > 0).all()
