"""Le2i GT có dòng trùng (video_id, frame_idx): mọi chỗ đọc GT dùng chung drop_duplicate_frames, giữ dòng đầu."""
import numpy as np
import pandas as pd

from src.data.common import drop_duplicate_frames
from src.data.labels import select_main_track
from src.models.m1_rule import gt_boxes_as_tracks, load_config, signals_by_track


def gt_with_duplicate(n_frames: int = 40) -> pd.DataFrame:
    """1 video Le2i, mỗi frame 1 hộp; frame 10 ghi thêm 1 dòng với hộp KHÁC, nằm sau trong file."""
    rows = [{"video_id": "le2i_home01_001", "frame_idx": f, "x1": 100.0, "y1": 20.0, "x2": 140.0, "y2": 160.0,
             "aux_code": 1} for f in range(n_frames)]
    rows.append({"video_id": "le2i_home01_001", "frame_idx": 10, "x1": 0.0, "y1": 0.0, "x2": 30.0, "y2": 30.0,
                 "aux_code": 1})
    return pd.DataFrame(rows)


def test_duplicate_frame_with_different_box_keeps_first_row():
    unique, dropped = drop_duplicate_frames(gt_with_duplicate())
    assert dropped == 1 and len(unique) == 40
    assert unique.loc[unique["frame_idx"] == 10, "x1"].tolist() == [100.0]


def test_gt_tracks_have_no_zero_time_step_in_signals(capsys):
    tracks = gt_boxes_as_tracks(gt_with_duplicate())
    assert "bỏ 1 dòng bbox trùng" in capsys.readouterr().out
    meta = pd.DataFrame([{"video_id": "le2i_home01_001", "fps": 25.0}])
    items = signals_by_track(tracks, meta, load_config("configs/m1.yaml"))
    for item in items:
        steps = np.diff(item["signals"].index.to_numpy())
        assert (steps > 0).all()  # không có dt = 0: mỗi frame đúng 1 dòng tín hiệu
        assert item["signals"].index.is_unique


def test_main_track_uses_first_gt_row_of_duplicated_frame():
    """Track 2 chỉ khớp hộp thứ hai (bị bỏ) nên không được chọn; giữ dòng đầu giống gt_boxes_as_tracks."""
    gt = gt_with_duplicate(n_frames=11).query("frame_idx == 10")
    tracks = pd.DataFrame([
        {"video_id": "le2i_home01_001", "frame_idx": 10, "track_id": 1, "x1": 100.0, "y1": 20.0, "x2": 140.0, "y2": 160.0},
        {"video_id": "le2i_home01_001", "frame_idx": 10, "track_id": 2, "x1": 0.0, "y1": 0.0, "x2": 30.0, "y2": 30.0},
    ])
    row = {"video_id": "le2i_home01_001", "dataset": "le2i", "width": 320, "height": 240}
    assert select_main_track(tracks, row, gt) == 1

