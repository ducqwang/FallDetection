"""Việc 4: 5 biểu đồ EDA cho báo cáo, lưu PNG dpi 200, nhãn tiếng Việt.

Đầu vào: metadata.csv cho biểu đồ 1-4; bbox (GT hoặc tracks cache) cho biểu đồ 5.
Đầu ra: reports/figures/eda_{1..5}_*.png. Biểu đồ 1-4 không đọc lại video.
plot_embedding nhận ma trận đặc trưng bất kỳ, sau này thay bằng keypoint hoặc embedding CNN không phải sửa code.
Ví dụ: python -m src.eda.plots --meta metadata.csv --gt-bbox annotations/le2i_gt_bbox.csv --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data.common import METADATA_DTYPES, drop_duplicate_frames
from ..data.frame_size import DEFAULT_WARNING_THRESHOLD, scene_size_table
from .embedding import plot_embedding, plot_embedding_pair  # noqa: F401  (plot_embedding giữ ở đây cho nơi gọi cũ)
from .size_check import measure_size_separation, update_data_card
from .style import (COLOR_ADL, COLOR_FALL, COLOR_HARD, DATASET_NAMES, _bar_labels, _save, plt)
from .window_features import build_window_features

def _labeled(meta: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ video has_event_label = 1."""
    return meta[meta["has_event_label"] == 1]


def plot_clip_counts(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 1: số clip theo lớp × nguồn, tách riêng ADL negative khó."""
    data = _labeled(meta)
    figure, axis = plt.subplots(figsize=(6.4, 4))
    positions, names = [], []
    for index, dataset in enumerate(DATASET_NAMES):
        part = data[data["dataset"] == dataset]
        n_fall = int((part["label"] == "fall").sum())
        n_hard = int(part["adl_hard_neg"].sum())
        n_adl = int((part["label"] == "adl").sum()) - n_hard
        fall_x, adl_x = index * 3, index * 3 + 1
        axis.bar(fall_x, n_fall, color=COLOR_FALL, width=0.8, label="Té ngã" if index == 0 else None)
        axis.bar(adl_x, n_adl, color=COLOR_ADL, width=0.8, label="ADL thường" if index == 0 else None)
        axis.bar(adl_x, n_hard, bottom=n_adl, color=COLOR_HARD, width=0.8, label="ADL negative khó" if index == 0 else None)
        _bar_labels(axis, [fall_x, adl_x, adl_x], [0, 0, n_adl], [n_fall, n_adl, n_hard])
        positions += [fall_x, adl_x]
        names += [f"{DATASET_NAMES[dataset]}\nTé ngã", f"{DATASET_NAMES[dataset]}\nADL"]
    axis.set_xticks(positions, names)
    axis.set_ylabel("Số clip")
    axis.grid(axis="x", visible=False)
    axis.set_title("Số clip theo lớp và nguồn", fontsize=10, loc="left")
    axis.legend()
    return _save(figure, out_path)


def plot_duration(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 2: phân bố duration_s (giây) theo lớp, mỗi nguồn một ô vì thang thời gian khác nhau."""
    data = _labeled(meta)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, dataset in zip(axes, DATASET_NAMES):
        part = data[data["dataset"] == dataset]
        bins = np.linspace(0, part["duration_s"].max() * 1.02, 16)
        for label, name, color in (("fall", "Té ngã", COLOR_FALL), ("adl", "ADL", COLOR_ADL)):
            values = part.loc[part["label"] == label, "duration_s"]
            axis.hist(values, bins=bins, histtype="step", linewidth=2, color=color,
                      label=f"{name}: trung vị {values.median():.1f} s (n={len(values)})")
        axis.set_xlabel("Độ dài clip (giây)")
        axis.set_ylabel("Số clip")
        axis.set_title(DATASET_NAMES[dataset], fontsize=10, loc="left")
        axis.legend(fontsize=8)
    return _save(figure, out_path)


def fall_frame_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """Đếm frame té ngã (trong [fall_start, fall_end], 0-based, tính cả hai đầu) và bình thường theo nguồn."""
    data = _labeled(meta).copy()
    data["fall_frames"] = (data["fall_end_frame"] - data["fall_start_frame"] + 1).fillna(0).astype(int)
    table = data.groupby("dataset").agg(fall=("fall_frames", "sum"), total=("n_frames", "sum"))
    table["normal"] = table["total"] - table["fall"]
    return table


def plot_frame_balance(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 3: tỷ lệ frame té ngã / bình thường, minh họa mất cân bằng lớp."""
    table = fall_frame_counts(meta)
    figure, axis = plt.subplots(figsize=(7.2, 2.8))
    for row_index, dataset in enumerate(DATASET_NAMES):
        fall, normal, total = table.loc[dataset, ["fall", "normal", "total"]]
        axis.barh(row_index, normal / total * 100, color=COLOR_ADL, height=0.55, label="Frame bình thường" if row_index == 0 else None)
        axis.barh(row_index, fall / total * 100, left=normal / total * 100, color=COLOR_FALL, height=0.55, label="Frame té ngã" if row_index == 0 else None)
        axis.text(50, row_index, f"té ngã {fall / total * 100:.1f}% ({int(fall)}/{int(total)} frame)", color="white", ha="center", va="center", fontsize=9)
    axis.set_yticks(range(len(DATASET_NAMES)), DATASET_NAMES.values())
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Tỷ lệ frame (%)")
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    return _save(figure, out_path)


def plot_scene_distribution(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 4: phân bố bối cảnh/camera; Le2i tách phần 01/02."""
    data = meta.copy()
    is_urfd = data["dataset"] == "urfd"
    data["group"] = (data["scene"].str.replace("_", " ").str.capitalize() + " " + data["scene_part"].fillna(""))
    data.loc[is_urfd, "group"] = "URFD " + data.loc[is_urfd, "camera"]
    groups = sorted(data["group"].unique(), key=lambda name: (not name.startswith("URFD"), name))
    figure, axis = plt.subplots(figsize=(7.2, 4))
    bottoms = np.zeros(len(groups))
    for label, name, color in (("fall", "Té ngã", COLOR_FALL), ("adl", "ADL", COLOR_ADL)):
        counts = data[data["label"] == label].groupby("group").size().reindex(groups, fill_value=0).to_numpy()
        axis.bar(groups, counts, bottom=bottoms, color=color, width=0.65, label=name)
        _bar_labels(axis, range(len(groups)), bottoms, counts)
        bottoms = bottoms + counts
    axis.set_ylabel("Số clip")
    axis.grid(axis="x", visible=False)
    axis.legend()
    return _save(figure, out_path)


def export_scene_size_table(meta: pd.DataFrame, out_csv: Path) -> Path:
    """Lưu bảng scene × kích thước × lớp của Le2i thành CSV."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scene_size_table(meta).to_csv(out_csv, index=False)
    return out_csv


def plot_scene_size_table(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 6: số video Le2i theo scene/phần × kích thước, chồng fall và ADL."""
    table = scene_size_table(meta)
    names = [f"{row.scene.replace('_', ' ')} {row.scene_part}\n{row.size}".replace("  ", " ") for row in table.itertuples()]
    figure, axis = plt.subplots(figsize=(7.2, 4))
    axis.bar(names, table["fall"], color=COLOR_FALL, width=0.65, label="Té ngã")
    axis.bar(names, table["adl"], bottom=table["fall"], color=COLOR_ADL, width=0.65, label="ADL")
    _bar_labels(axis, range(len(names)), np.zeros(len(names)), table["fall"].to_numpy())
    _bar_labels(axis, range(len(names)), table["fall"].to_numpy(), table["adl"].to_numpy())
    axis.set_ylabel("Số video Le2i")
    axis.grid(axis="x", visible=False)
    axis.set_title("Le2i: scene × kích thước khung × lớp", fontsize=10, loc="left")
    axis.legend()
    return _save(figure, out_path)


def main() -> None:
    """Vẽ các biểu đồ EDA, xuất bảng kích thước khung và ghi cảnh báo kích thước vào DATA_CARD nếu có --data-card."""
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ EDA (1-5, 5b theo kích thước, 6 bảng kích thước).")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--out-dir", default="reports/figures", help="thư mục lưu PNG")
    parser.add_argument("--gt-bbox", required=True, help="le2i_gt_bbox.csv")
    parser.add_argument("--tracks", help="tracks.parquet; có thì thay GT làm nguồn bbox của biểu đồ 5")
    parser.add_argument("--config", default="configs/m1.yaml", help="lấy tham số tín hiệu từ đây")
    parser.add_argument("--window-s", type=float, default=1.0, help="độ dài cửa sổ (giây)")
    parser.add_argument("--stride-s", type=float, default=0.25, help="bước trượt (giây)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="perplexity của t-SNE")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--results-dir", default="reports/results", help="thư mục lưu CSV bảng kích thước và độ tách")
    parser.add_argument("--data-card", help="DATA_CARD.md; có thì ghi bảng kích thước và cảnh báo độ tách vào khối riêng")
    parser.add_argument("--size-warn", type=float, default=DEFAULT_WARNING_THRESHOLD,
                        help="ngưỡng silhouette theo kích thước để ghi cảnh báo")
    args = parser.parse_args()
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    out_dir = Path(args.out_dir)
    written = [plot_clip_counts(meta, out_dir / "eda_1_clip_counts.png"), plot_duration(meta, out_dir / "eda_2_duration.png"),
               plot_frame_balance(meta, out_dir / "eda_3_frame_balance.png"),
               plot_scene_distribution(meta, out_dir / "eda_4_scene_distribution.png")]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gt_boxes, n_dropped = drop_duplicate_frames(pd.read_csv(args.gt_bbox))
    if n_dropped:
        print(f"CẢNH BÁO | Le2i GT: bỏ {n_dropped} dòng bbox trùng (video_id, frame_idx), giữ dòng đầu")
    # Ghi bảng kích thước và khối DATA_CARD trước phần đặc trưng: chúng chỉ cần metadata nên không phụ thuộc bbox.
    written.append(plot_scene_size_table(meta, out_dir / "eda_6_le2i_scene_size.png"))
    written.append(export_scene_size_table(meta, Path(args.results_dir) / "le2i_scene_size_table.csv"))
    if args.data_card:
        update_data_card(args.data_card, meta, None, args.size_warn)
    boxes = pd.read_parquet(args.tracks) if args.tracks else gt_boxes
    X, y, names, sizes = build_window_features(boxes, meta, args.window_s, args.stride_s, config["signals"],
                                               config["rules"]["still_speed"], gt_boxes if args.tracks else None, with_size=True)
    source = f"{len(names)} đặc trưng hình học, cửa sổ {args.window_s:g} s, bbox từ {'tracks cache' if args.tracks else 'Le2i GT'}"
    written += plot_embedding_pair(X, y, sizes, source, out_dir, args.perplexity, args.seed)
    in_le2i = np.isin(sizes, scene_size_table(meta)["size"].unique())  # độ tách chỉ đo trên các kích thước của Le2i
    separation = None
    if len(set(sizes[in_le2i])) >= 2:
        separation = measure_size_separation(X[in_le2i], y[in_le2i], sizes[in_le2i], args.seed)
        out_json = Path(args.results_dir) / "size_separation.json"
        out_json.write_text(json.dumps(separation, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_json)
        print(f"ĐỘ TÁCH THEO KÍCH THƯỚC | {separation}")
    else:
        print("ĐỘ TÁCH THEO KÍCH THƯỚC | bỏ qua: cửa sổ chỉ có 1 kích thước khung")
    if args.data_card:
        written.append(update_data_card(args.data_card, meta, separation, args.size_warn))
    n_unlabeled = int((meta["has_event_label"] == 0).sum())
    print(f"TÓM TẮT | video trong metadata: {len(meta)} | cửa sổ: {len(y)} (té ngã {int(y.sum())}) "
          f"| bỏ qua ở biểu đồ 1-3, 5: {n_unlabeled} (has_event_label = 0) | Đã ghi: "
          + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
