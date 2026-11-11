"""Test validate_tracks (src/data/tracks.py): mỗi loại vi phạm một test, cộng trường hợp hợp lệ.

Ví dụ: python -m pytest tests/test_tracks_validate.py -q
"""
import pandas as pd
import pytest
from conftest import meta_frame, meta_row

from src.data.tracks import cast_tracks, clip_to_frame, validate_tracks

META = meta_frame([meta_row("v1", "adl", n_frames=10), meta_row("v2", "adl", n_frames=10),
                   meta_row("v_empty", "adl", n_frames=10)])  # khung 640×480


def good_tracks() -> pd.DataFrame:
    rows = [(video, frame, 1, 10.0, 20.0, 110.0, 220.0, 0.9) for video in ("v1", "v2") for frame in range(5)]
    return cast_tracks(pd.DataFrame(rows, columns=["video_id", "frame_idx", "track_id", "x1", "y1", "x2", "y2",
                                                   "det_conf"]))


def codes_after(change) -> set[str]:
    tracks = good_tracks()
    change(tracks)
    return validate_tracks(tracks, META).codes()


def test_valid_tracks_pass_and_list_empty_videos():
    check = validate_tracks(good_tracks(), META)
    assert check.ok and check.empty_videos == ["v_empty"]
    assert "v_empty" in check.summary()


def test_missing_column():
    check = validate_tracks(good_tracks().drop(columns="det_conf"), META)
    assert check.codes() == {"columns"}


def test_unknown_video_id():
    assert codes_after(lambda t: t.loc.__setitem__((0, "video_id"), "lạ")) == {"unknown_video"}


@pytest.mark.parametrize("frame", [-1, 10, 25])
def test_frame_idx_outside_range(frame):
    assert codes_after(lambda t: t.loc.__setitem__((0, "frame_idx"), frame)) == {"frame_range"}


def test_duplicate_key():
    tracks = good_tracks()
    duplicated = pd.concat([tracks, tracks.iloc[[2]]], ignore_index=True)
    assert validate_tracks(duplicated, META).codes() == {"duplicate"}


@pytest.mark.parametrize("column, value", [("x1", -0.5), ("y1", -3.0), ("x2", 640.5), ("y2", 481.0)])
def test_box_outside_frame(column, value):
    assert codes_after(lambda t: t.loc.__setitem__((0, column), value)) == {"box_outside"}


@pytest.mark.parametrize("column, value", [("x2", 10.0), ("x2", 5.0), ("y2", 20.0), ("y2", 1.0)])
def test_box_without_area(column, value):
    assert codes_after(lambda t: t.loc.__setitem__((0, column), value)) == {"box_area"}


def test_wrong_dtypes():
    def float_frames(tracks):
        tracks["frame_idx"] = tracks["frame_idx"].astype(float)
    assert codes_after(float_frames) == {"dtype"}

    def int_boxes(tracks):
        tracks["x1"] = tracks["x1"].astype("int64")
    assert codes_after(int_boxes) == {"dtype"}

    def numeric_video(tracks):
        tracks["video_id"] = 7
    assert "dtype" in codes_after(numeric_video)


def test_det_conf_outside_unit_interval():
    assert codes_after(lambda t: t.loc.__setitem__((0, "det_conf"), 1.5)) == {"det_conf"}


def test_clip_to_frame_counts_dropped_boxes():
    tracks = good_tracks().head(3).copy()
    tracks.loc[0, ["x1", "x2"]] = [-50.0, 700.0]
    tracks.loc[1, ["x1", "x2"]] = [650.0, 700.0]
    clipped, n_dropped = clip_to_frame(tracks, 640, 480)
    assert n_dropped == 1 and clipped.loc[0, ["x1", "x2"]].tolist() == [0.0, 640.0]
