"""Test bảng scene × kích thước × lớp, color_by của plot_embedding, độ tách theo kích thước và khối DATA_CARD (dữ liệu giả)."""
import numpy as np
import pandas as pd
import pytest

from src.data.frame_size import (BLOCK_END, BLOCK_START, replace_block, scene_size_table, size_block,
                                 single_class_sizes, static_warning)
from src.data.data_card import build_data_card
from src.eda.embedding import plot_embedding, plot_embedding_pair
from src.eda.plots import export_scene_size_table, plot_scene_size_table
from src.eda.size_check import measure_size_separation, update_data_card
from src.eda.window_features import build_window_features
from conftest import meta_frame, meta_row


def le2i_meta() -> pd.DataFrame:
    """Le2i giả: home01 có 3 fall 320×180 và 2 fall 320×240; home02 có 1 fall và 2 ADL, đều 320×240."""
    rows = []
    cells = [("home", "01", "fall", 320, 180, 3), ("home", "01", "fall", 320, 240, 2),
             ("home", "02", "fall", 320, 240, 1), ("home", "02", "adl", 320, 240, 2)]
    number = 0
    for scene, part, label, width, height, count in cells:
        for _ in range(count):
            number += 1
            row = meta_row(f"le2i_{scene}{part}_{number:03d}", label, dataset="le2i", scene=scene, fps=25.0,
                           start=10 if label == "fall" else None, end=20 if label == "fall" else None)
            rows.append({**row, "scene_part": part, "width": width, "height": height})
    rows.append(meta_row("urfd_fall-01_cam0", "fall", start=50, end=80))
    return meta_frame(rows)


def test_table_counts_scene_size_and_class():
    table = scene_size_table(le2i_meta())
    cell = table.set_index(["scene_part", "size"])
    assert cell.loc[("01", "320×180"), ["fall", "adl"]].tolist() == [3, 0]
    assert cell.loc[("01", "320×240"), ["fall", "adl"]].tolist() == [2, 0]
    assert cell.loc[("02", "320×240"), ["fall", "adl"]].tolist() == [1, 2]
    assert table["total"].sum() == 8  # không đếm URFD


def test_table_ignores_unlabeled_videos():
    meta = le2i_meta()
    meta.loc[meta["video_id"] == "le2i_home01_001", "has_event_label"] = 0
    assert scene_size_table(meta)["total"].sum() == 7


def test_single_class_size_is_flagged_in_warning():
    table = scene_size_table(le2i_meta())
    assert single_class_sizes(table) == ["320×180"]
    warning = static_warning(table)
    assert "3 video 320×180" in warning and "tất cả đều là fall" in warning and "toạ độ pixel" in warning


def test_no_warning_when_every_size_has_both_classes():
    only_240 = scene_size_table(le2i_meta()).query("size == '320×240'")
    assert single_class_sizes(only_240) == []
    assert "Cảnh báo" not in static_warning(only_240)


def test_csv_and_figure_are_written(tmp_path):
    meta = le2i_meta()
    csv = export_scene_size_table(meta, tmp_path / "r" / "t.csv")
    assert pd.read_csv(csv, dtype={"scene_part": str}).shape == (3, 6)
    assert plot_scene_size_table(meta, tmp_path / "t.png").stat().st_size > 0


def clouds(shift: float, n: int = 120, seed: int = 0):
    """X 4 chiều: nửa đầu kích thước A, nửa sau B bị dịch `shift`; nhãn lẫn đều trong cả hai nhóm."""
    rng = np.random.default_rng(seed)
    sizes = np.array(["320×180"] * n + ["320×240"] * n)
    X = rng.normal(size=(2 * n, 4))
    X[n:] += shift
    y = np.tile([0, 1], n)
    return X, y, sizes


def test_separation_high_when_size_shifts_features_and_low_when_not():
    high = measure_size_separation(*clouds(shift=6.0))
    low = measure_size_separation(*clouds(shift=0.0))
    assert high["silhouette_size"] > 0.25 > low["silhouette_size"]
    assert high["centroid_distance"] > low["centroid_distance"]
    assert high["silhouette_size_fall"] > 0.25 > low["silhouette_size_fall"]


def test_separation_needs_two_sizes():
    X, y, sizes = clouds(0.0)
    with pytest.raises(ValueError, match="2 kích thước"):
        measure_size_separation(X, y, np.array(["320×240"] * len(y)))
    with pytest.raises(ValueError, match="khác độ dài"):
        measure_size_separation(X, y[:-1], sizes)


def test_color_by_size_draws_one_group_per_size_and_needs_sizes(tmp_path):
    X, y, sizes = clouds(shift=3.0, n=60)
    out = plot_embedding(X, y, "test", tmp_path / "s.png", perplexity=10, color_by="size", sizes=sizes)
    assert out.stat().st_size > 0
    with pytest.raises(ValueError, match="sizes"):
        plot_embedding(X, y, "test", tmp_path / "x.png", color_by="size")
    with pytest.raises(ValueError, match="color_by"):
        plot_embedding(X, y, "test", tmp_path / "x.png", color_by="scene", sizes=sizes)
    with pytest.raises(ValueError, match="sizes có"):
        plot_embedding(X, y, "test", tmp_path / "x.png", color_by="size", sizes=sizes[:-1])


def test_default_color_by_label_is_unchanged(tmp_path):
    X, y, _ = clouds(shift=0.0, n=60)
    assert plot_embedding(X, y, "test", tmp_path / "l.png", perplexity=10).exists()


def test_pair_writes_both_versions(tmp_path):
    X, y, sizes = clouds(shift=3.0, n=60)
    paths = plot_embedding_pair(X, y, sizes, "test", tmp_path, perplexity=10)
    assert [p.name for p in paths] == ["eda_5_pca_tsne.png", "eda_5b_pca_tsne_by_size.png"]
    assert all(p.exists() for p in paths)


def test_window_features_return_size_per_window():
    meta = le2i_meta()
    rows = []
    for video_id in meta.loc[meta["dataset"] == "le2i", "video_id"]:
        for frame in range(60):
            rows.append({"video_id": video_id, "frame_idx": frame, "x1": 50.0, "y1": 20.0 + frame, "x2": 90.0, "y2": 120.0 + frame})
    boxes = pd.DataFrame(rows)
    params = {"smooth_s": 0.2, "reference_s": 1.5, "max_gap_s": 0.5}
    X, y, names, sizes = build_window_features(boxes, meta, 0.5, 0.25, params, 0.15, with_size=True)
    assert len(sizes) == len(y) == len(X) and set(sizes) == {"320×180", "320×240"}
    assert len(build_window_features(boxes, meta, 0.5, 0.25, params, 0.15)) == 3  # mặc định không đổi


def test_data_card_block_is_replaced_not_duplicated(tmp_path):
    meta = le2i_meta()
    card = tmp_path / "DATA_CARD.md"
    card.write_text("# DATA CARD\n\nphần khác\n", encoding="utf-8")
    update_data_card(card, meta, None)
    high = measure_size_separation(*clouds(shift=6.0))
    update_data_card(card, meta, high)
    text = card.read_text(encoding="utf-8")
    assert text.count(BLOCK_START) == 1 and text.count(BLOCK_END) == 1 and "phần khác" in text
    assert "hai nhóm kích thước tách nhau rõ" in text and "3 video 320×180" in text
    update_data_card(card, meta, measure_size_separation(*clouds(shift=0.0)))
    assert "hai nhóm kích thước tách nhau rõ" not in card.read_text(encoding="utf-8")


def test_replace_block_appends_when_missing():
    assert replace_block("abc\n", f"{BLOCK_START}\nx\n{BLOCK_END}\n").endswith(f"{BLOCK_END}\n")


def test_build_data_card_contains_size_block():
    meta = le2i_meta()
    text = build_data_card(meta, [], [], {"n_rows": 0, "n_zero": 0, "n_degenerate": 0}, {}, [], 15)
    assert BLOCK_START in text and "320×180" in text


def test_cli_writes_figures_tables_and_data_card(tmp_path):
    """Chạy cả CLI plots không cần ffmpeg: metadata và GT bbox giả, đủ cửa sổ fall và non-fall ở cả 2 kích thước."""
    import subprocess
    import sys
    from pathlib import Path

    meta = le2i_meta().assign(camera=lambda t: t["dataset"].map({"urfd": "cam0"}))
    meta.loc[meta["dataset"] == "urfd", "scene_part"] = pd.NA
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    rows = []
    for number, video_id in enumerate(meta.loc[meta["dataset"] == "le2i", "video_id"]):
        for frame in range(90):
            drop = 40.0 * max(0, min(frame - 10, 10)) / 10  # hộp bẹt dần trong khoảng 10-20 như lúc ngã
            rows.append({"video_id": video_id, "frame_idx": frame, "x1": 50.0 + number, "y1": 20.0 + drop,
                         "x2": 90.0 + drop, "y2": 120.0})
    rows += [{**rows[3], "x1": 60.0}, dict(rows[95])]  # frame ghi trùng: 1 dòng khác hộp, 1 dòng y hệt (như Le2i GT thật)
    gt = tmp_path / "gt.csv"
    pd.DataFrame(rows).to_csv(gt, index=False)
    card = tmp_path / "DATA_CARD.md"
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, "-m", "src.eda.plots", "--meta", str(meta_path), "--gt-bbox", str(gt),
                           "--out-dir", str(tmp_path / "fig"), "--results-dir", str(tmp_path / "res"),
                           "--data-card", str(card), "--window-s", "0.5", "--perplexity", "10"],
                          cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "fig" / "eda_5b_pca_tsne_by_size.png").exists() and (tmp_path / "fig" / "eda_6_le2i_scene_size.png").exists()
    assert (tmp_path / "res" / "le2i_scene_size_table.csv").exists() and (tmp_path / "res" / "size_separation.json").exists()
    assert "3 video 320×180" in card.read_text(encoding="utf-8")
    assert "bỏ 2 dòng bbox trùng" in done.stdout


def test_duplicate_gt_frames_keep_first_and_are_counted():
    from src.models.m1_features import drop_duplicate_frames
    from src.models.m1_rule import gt_boxes_as_tracks
    boxes = pd.DataFrame({"video_id": ["v"] * 4, "frame_idx": [0, 1, 1, 2], "x1": [1.0, 2, 9, 3], "y1": [1.0] * 4,
                          "x2": [5.0] * 4, "y2": [5.0] * 4, "aux_code": [1] * 4})
    unique, dropped = drop_duplicate_frames(boxes)
    assert dropped == 1 and unique["frame_idx"].tolist() == [0, 1, 2] and unique.loc[1, "x1"] == 2
    assert gt_boxes_as_tracks(boxes)["frame_idx"].is_unique


def test_size_table_is_written_even_if_window_features_fail(tmp_path):
    """Bảng kích thước chỉ cần metadata nên phải có dù bước đặc trưng cửa sổ lỗi (bbox rỗng)."""
    import subprocess
    import sys
    from pathlib import Path
    meta = le2i_meta().assign(camera="cam0")
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    gt = tmp_path / "gt.csv"
    pd.DataFrame(columns=["video_id", "frame_idx", "x1", "y1", "x2", "y2", "aux_code"]).to_csv(gt, index=False)
    subprocess.run([sys.executable, "-m", "src.eda.plots", "--meta", str(meta_path), "--gt-bbox", str(gt),
                    "--out-dir", str(tmp_path / "fig"), "--results-dir", str(tmp_path / "res")],
                   cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
    assert (tmp_path / "res" / "le2i_scene_size_table.csv").exists()
