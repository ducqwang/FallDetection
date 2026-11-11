"""Test báo cáo chất lượng detector (src/eval/track_quality.py, scripts/check_tracks_quality.py) trên dữ liệu giả.

Ví dụ: python -m pytest tests/test_track_quality.py -q
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import meta_frame, meta_row

from src.data.common import TRACK_COLUMNS
from src.eval.track_quality import le2i_track_quality, summarize, urfd_track_quality

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_tracks_quality  # noqa: E402

BOX = (10.0, 10.0, 110.0, 210.0)


def le2i_meta():
    rows = [meta_row("le2i_home01_001", "fall", dataset="le2i", scene="home", n_frames=10, start=3, end=6),
            meta_row("le2i_home02_002", "adl", dataset="le2i", scene="home", n_frames=10),
            meta_row("le2i_coffee_room01_003", "fall", dataset="le2i", scene="coffee_room", n_frames=10, start=3, end=6),
            meta_row("urfd_adl-01_cam0", "adl", n_frames=10)]
    return meta_frame(rows)


def gt_rows(video_id, frames, box=BOX):
    return [(video_id, frame, *box, 0) for frame in frames]


def track_rows(video_id, track_id, frames, box=BOX):
    return [(video_id, frame, track_id, *box, 0.9) for frame in frames]


@pytest.fixture
def inputs():
    shifted = (60.0, 10.0, 160.0, 210.0)  # IoU với BOX = 50·200 / (2·100·200 − 50·200) = 1/3
    gt = pd.DataFrame(gt_rows("le2i_home01_001", range(10)) + gt_rows("le2i_home01_001", [0], shifted)
                      + gt_rows("le2i_home02_002", range(10)) + gt_rows("le2i_coffee_room01_003", range(4)),
                      columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2", "aux_code"])
    tracks = pd.DataFrame(track_rows("le2i_home01_001", 1, range(10)) + track_rows("le2i_home01_001", 2, range(3), shifted)
                          + track_rows("le2i_home02_002", 5, range(5)) + track_rows("le2i_home02_002", 6, range(5, 10), shifted)
                          + track_rows("urfd_adl-01_cam0", 1, range(4)) + track_rows("urfd_adl-01_cam0", 2, range(2)),
                          columns=TRACK_COLUMNS)
    return tracks, le2i_meta(), gt


def test_le2i_main_track_iou_and_hit_rate(inputs):
    per_video = le2i_track_quality(*inputs).set_index("video_id")
    perfect = per_video.loc["le2i_home01_001"]  # dòng GT trùng frame 0 bị dedupe, giữ hộp đầu
    assert perfect["main_track"] == 1 and perfect["n_gt_frames"] == 10
    assert perfect["mean_iou"] == pytest.approx(1.0) and perfect["hit_rate"] == 1.0
    half = per_video.loc["le2i_home02_002"]  # track 5 khớp 5/10 frame, frame thiếu tính IoU 0
    assert half["main_track"] == 5 and half["main_coverage"] == 0.5
    assert half["mean_iou"] == pytest.approx(0.5) and half["hit_rate"] == 0.5
    missing = per_video.loc["le2i_coffee_room01_003"]  # không có track nào
    assert missing["n_tracks"] == 0 and pd.isna(missing["main_track"]) and missing["hit_rate"] == 0.0


def test_summaries_by_scene_and_label(inputs):
    per_video = le2i_track_quality(*inputs)
    by_scene = summarize(per_video, "scene").set_index("scene")
    assert by_scene.loc["home", "n_videos"] == 2 and by_scene.loc["home", "hit_rate"] == pytest.approx(0.75)
    assert by_scene.loc["all", "n_videos"] == 3 and by_scene.loc["all", "hit_rate"] == pytest.approx(0.5)
    by_label = summarize(per_video, "label").set_index("label")
    assert by_label.loc["fall", "hit_rate"] == pytest.approx(0.5) and by_label.loc["adl", "hit_rate"] == 0.5


def test_urfd_track_count_and_frame_fraction(inputs):
    tracks, meta, _ = inputs
    row = urfd_track_quality(tracks, meta).iloc[0]
    assert row["n_tracks"] == 2 and row["frac_frames_with_track"] == pytest.approx(0.4)


def test_cli_writes_csvs_for_processed_videos_only(inputs, tmp_path):
    tracks, meta, gt = inputs
    meta.assign(path="videos/x.mp4").to_csv(tmp_path / "metadata.csv", index=False)
    (tmp_path / "annotations").mkdir()
    gt.to_csv(tmp_path / "annotations" / "le2i_gt_bbox.csv", index=False)
    tracks.to_parquet(tmp_path / "tracks.parquet", index=False)
    processed = ["le2i_home01_001", "le2i_coffee_room01_003", "urfd_adl-01_cam0"]
    (tmp_path / "tracks_meta.json").write_text(json.dumps({"videos": [{"video_id": v} for v in processed]}))
    assert check_tracks_quality.main(["--tracks", str(tmp_path / "tracks.parquet"), "--root", str(tmp_path)]) == 0
    out = tmp_path / "quality"
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "le2i_track_quality.csv", "le2i_track_quality_by_scene.csv", "le2i_track_quality_by_label.csv",
        "urfd_track_quality.csv", "urfd_track_quality_by_label.csv"])
    assert set(pd.read_csv(out / "le2i_track_quality.csv")["video_id"]) == {"le2i_home01_001", "le2i_coffee_room01_003"}
