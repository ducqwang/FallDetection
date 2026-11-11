"""Test scripts/make_tracks.py bằng model giả (không tải trọng số, không cần ultralytics/GPU).

FakeModel mô phỏng model.track() của ultralytics: trả từng frame một kết quả có boxes.xyxy/conf/id;
tracker giữ bảng người→id và bộ đếm id, chỉ làm mới khi persist=False (hoặc khi reset()).
Ví dụ: python -m pytest tests/test_make_tracks.py -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import meta_frame, meta_row

from src.data.common import PREDICTION_COLUMNS, TRACK_COLUMNS
from src.data.loader import load_tracks
from src.data.track_runner import reset_trackers
from src.data.tracks import validate_tracks
from src.models import m1_rule

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_tracks  # noqa: E402

STAND = (100.0, 100.0, 160.0, 270.0)


class FakeTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ids, self.next_id = {}, 1

    def assign(self, person: str) -> int:
        if person not in self.ids:
            self.ids[person], self.next_id = self.next_id, self.next_id + 1
        return self.ids[person]


class FakeModel:
    """videos: {video_id: [frame, ...]}, frame = list (người hoặc None, hộp, conf). Frame toàn None: chưa gán id.

    honor_persist=False mô phỏng trường hợp xấu: tracker cũ luôn được giữ, chỉ reset_trackers mới làm mới.
    """

    def __init__(self, videos: dict, honor_persist: bool = True, forbidden: tuple = ()):
        self.videos, self.honor_persist, self.forbidden = videos, honor_persist, forbidden
        self.predictor, self.calls, self.ckpt_path = None, [], None

    def track(self, source, persist, **kwargs):
        video_id = Path(source).stem
        assert video_id not in self.forbidden, f"{video_id} lẽ ra phải được bỏ qua nhờ --resume"
        self.calls.append({"video_id": video_id, "persist": persist, **kwargs})
        if self.predictor is None or (self.honor_persist and not persist):
            self.predictor = SimpleNamespace(trackers=[FakeTracker()])
        tracker = self.predictor.trackers[0]
        for frame in self.videos[video_id]:
            people = [item[0] for item in frame]
            ids = None if frame and all(p is None for p in people) else np.array([tracker.assign(p) for p in people])
            yield SimpleNamespace(boxes=SimpleNamespace(xyxy=np.array([item[1] for item in frame], float).reshape(-1, 4),
                                                        conf=np.array([item[2] for item in frame], float), id=ids))


def person(name="a", box=STAND, conf=0.9):
    return [(name, box, conf)]


def make_dataset(tmp_path: Path, frames: dict, n_frames: dict | None = None) -> Path:
    """Ghi metadata.csv + file video rỗng (FakeModel không đọc file) cho các video URFD giả."""
    rows = []
    (tmp_path / "videos").mkdir()
    for video_id, video_frames in frames.items():
        (tmp_path / "videos" / f"{video_id}.mp4").touch()
        count = (n_frames or {}).get(video_id, len(video_frames))
        rows.append({**meta_row(video_id, "adl", n_frames=count), "path": f"videos/{video_id}.mp4"})
    meta_frame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path


def run(root: Path, model: FakeModel, *extra: str) -> tuple[int, Path]:
    out_dir = root / "out"
    code = make_tracks.run(make_tracks.parse_args(["--root", str(root), "--out-dir", str(out_dir), "--device", "cpu",
                                                   *extra]), model=model)
    return code, out_dir


def sidecar(out_dir: Path) -> dict:
    return json.loads((out_dir / "tracks_meta.json").read_text(encoding="utf-8"))


def test_output_has_exact_schema_and_dtypes(tmp_path):
    videos = {"urfd_adl-01_cam0": [person()] * 5, "urfd_adl-02_cam0": [person("b")] * 4}
    code, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos))
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    assert code == 0
    assert list(tracks.columns) == TRACK_COLUMNS
    assert tracks["frame_idx"].dtype == "int64" and tracks["track_id"].dtype == "int64"
    assert all(tracks[c].dtype == "float64" for c in ["x1", "y1", "x2", "y2", "det_conf"])
    assert len(tracks) == 9 and validate_tracks(tracks, pd.read_csv(tmp_path / "metadata.csv")).ok
    assert list(tracks.index) == list(tracks.sort_values(["video_id", "frame_idx", "track_id"]).index)


def test_frame_idx_is_read_order_from_zero_including_empty_frames(tmp_path):
    videos = {"urfd_adl-01_cam0": [person(), [], person(), person()]}
    _, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos))
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    assert tracks["frame_idx"].tolist() == [0, 2, 3]


@pytest.mark.parametrize("honor_persist", [True, False])
def test_tracker_resets_between_videos(tmp_path, honor_persist):
    videos = {"urfd_adl-01_cam0": [person("a") + person("b", (300.0, 50.0, 350.0, 200.0))] * 3,
              "urfd_adl-02_cam0": [person("c")] * 3}
    model = FakeModel(videos, honor_persist=honor_persist)
    _, out_dir = run(make_dataset(tmp_path, videos), model)
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    first = tracks[tracks["video_id"] == "urfd_adl-01_cam0"]
    second = tracks[tracks["video_id"] == "urfd_adl-02_cam0"]
    assert sorted(first["track_id"].unique()) == [1, 2]
    assert second["track_id"].unique().tolist() == [1], "track_id phải bắt đầu lại từ 1, không nối tiếp video trước"
    assert all(call["persist"] is False for call in model.calls)
    assert all(call["classes"] == [0] for call in model.calls)


def test_reset_trackers_clears_state_left_in_predictor():
    model = FakeModel({})
    model.predictor = SimpleNamespace(trackers=[FakeTracker()])
    model.predictor.trackers[0].assign("a")
    reset_trackers(model)
    assert model.predictor.trackers[0].assign("z") == 1


def test_detections_without_track_id_are_dropped_and_counted(tmp_path):
    no_id = [(None, STAND, 0.4), (None, (10.0, 10.0, 50.0, 90.0), 0.3)]
    videos = {"urfd_adl-01_cam0": [no_id, person(), no_id, person()]}
    code, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos))
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    assert code == 0 and tracks["frame_idx"].tolist() == [1, 3]
    assert sidecar(out_dir)["dropped"]["n_no_id"] == 4


def test_boxes_clipped_to_frame_and_zero_area_dropped(tmp_path):
    frame = [("a", (-20.0, 400.0, 100.0, 520.0), 0.9), ("b", (700.0, 10.0, 800.0, 100.0), 0.8)]  # khung 640×480
    videos = {"urfd_adl-01_cam0": [frame] * 2}
    _, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos))
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    assert tracks[["x1", "y1", "x2", "y2"]].iloc[0].tolist() == [0.0, 400.0, 100.0, 480.0]
    assert len(tracks) == 2 and sidecar(out_dir)["dropped"]["n_zero_area"] == 2


def test_resume_skips_existing_shards_and_merges_all(tmp_path):
    videos = {"urfd_adl-01_cam0": [person()] * 3, "urfd_adl-02_cam0": [person()] * 2}
    root = make_dataset(tmp_path, videos)
    code, out_dir = run(root, FakeModel(videos), "--limit", "1")
    assert code == 0 and (out_dir / "shards" / "urfd_adl-01_cam0.parquet").is_file()
    model = FakeModel(videos, forbidden=("urfd_adl-01_cam0",))
    code, _ = run(root, model)
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    assert code == 0 and [call["video_id"] for call in model.calls] == ["urfd_adl-02_cam0"]
    assert tracks.groupby("video_id").size().to_dict() == {"urfd_adl-01_cam0": 3, "urfd_adl-02_cam0": 2}
    assert [item["resumed"] for item in sidecar(out_dir)["videos"]] == [True, False]


def test_resume_reruns_broken_shard_or_other_parameters(tmp_path):
    videos = {"urfd_adl-01_cam0": [person()] * 3, "urfd_adl-02_cam0": [person()] * 2}
    root = make_dataset(tmp_path, videos)
    _, out_dir = run(root, FakeModel(videos))
    (out_dir / "shards" / "urfd_adl-01_cam0.parquet").write_bytes(b"hong")  # shard dở do Kaggle ngắt
    model = FakeModel(videos)
    run(root, model)
    assert [call["video_id"] for call in model.calls] == ["urfd_adl-01_cam0"]
    model = FakeModel(videos)
    run(root, model, "--conf", "0.5")
    assert len(model.calls) == 2
    model = FakeModel(videos)
    run(root, model, "--conf", "0.5", "--no-resume")
    assert len(model.calls) == 2


def test_frame_count_off_by_one_is_warning_and_extra_frame_dropped(tmp_path, capsys):
    videos = {"urfd_adl-01_cam0": [person()] * 4, "urfd_adl-02_cam0": [person()] * 4}
    root = make_dataset(tmp_path, videos, n_frames={"urfd_adl-01_cam0": 3, "urfd_adl-02_cam0": 5})
    code, out_dir = run(root, FakeModel(videos))
    info = sidecar(out_dir)
    assert code == 0
    assert info["warnings"] == ["urfd_adl-01_cam0: đọc được 4 frame, metadata 3",
                                "urfd_adl-02_cam0: đọc được 4 frame, metadata 5"]
    assert info["dropped"]["n_out_of_range"] == 1
    assert pd.read_parquet(out_dir / "tracks.parquet")["frame_idx"].max() == 3
    assert "urfd_adl-01_cam0: đọc được 4 frame, metadata 3" in capsys.readouterr().out


def test_frame_count_off_by_more_than_one_is_error(tmp_path):
    videos = {"urfd_adl-01_cam0": [person()] * 4, "urfd_adl-02_cam0": [person()] * 4}
    root = make_dataset(tmp_path, videos, n_frames={"urfd_adl-02_cam0": 6})
    code, out_dir = run(root, FakeModel(videos))
    assert code == 1 and not (out_dir / "tracks.parquet").exists()
    assert sidecar(out_dir)["errors"] == ["urfd_adl-02_cam0: đọc được 4 frame, metadata 6"]
    assert (out_dir / "shards" / "urfd_adl-01_cam0.parquet").is_file()


def test_video_without_people_is_valid_and_reported(tmp_path):
    videos = {"urfd_adl-01_cam0": [[]] * 3, "urfd_adl-02_cam0": [person()] * 2}
    code, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos))
    assert code == 0 and sidecar(out_dir)["videos_without_tracks"] == ["urfd_adl-01_cam0"]
    assert pd.read_parquet(out_dir / "shards" / "urfd_adl-01_cam0.parquet").empty


def test_sidecar_records_parameters_versions_and_counts(tmp_path):
    videos = {"urfd_adl-01_cam0": [person()] * 3}
    _, out_dir = run(make_dataset(tmp_path, videos), FakeModel(videos), "--model", "yolo11m.pt")
    info = sidecar(out_dir)
    assert info["args"]["model"] == "yolo11m.pt" and info["args"]["conf"] == 0.25 and info["args"]["resume"] is True
    for key in ("ultralytics", "torch", "opencv", "git_commit", "tracker_yaml", "n_videos", "n_rows", "dropped",
                "warnings", "run_seconds", "videos_without_tracks"):
        assert key in info, key
    assert info["n_videos"] == 1 and info["n_rows"] == 3


def test_m1_rule_reads_generated_tracks(tmp_path):
    lying = (60.0, 300.0, 230.0, 360.0)
    videos = {"urfd_adl-01_cam0": [person()] * 60 + [person(box=lying)] * 60}
    root = make_dataset(tmp_path, videos)
    _, out_dir = run(root, FakeModel(videos))
    tracks = load_tracks(out_dir / "tracks.parquet")
    meta = pd.read_csv(root / "metadata.csv")
    predictions = m1_rule.predict(tracks, meta, m1_rule.load_config("configs/m1.yaml"))
    assert list(predictions.columns) == PREDICTION_COLUMNS and len(predictions) == 120
