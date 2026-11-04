"""Việc 4: 5 biểu đồ EDA cho báo cáo, lưu PNG dpi 200, nhãn tiếng Việt.

Đầu vào: metadata.csv cho biểu đồ 1-4; bbox (GT hoặc tracks cache) cho biểu đồ 5.
Đầu ra: reports/figures/eda_{1..5}_*.png. Biểu đồ 1-4 không đọc lại video.
plot_embedding nhận ma trận đặc trưng bất kỳ, sau này thay bằng keypoint hoặc embedding CNN không phải sửa code.
Ví dụ: python -m src.eda.plots --meta metadata.csv --gt-bbox annotations/le2i_gt_bbox.csv --out-dir reports/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ..data.common import METADATA_DTYPES  # noqa: E402
from .window_features import build_window_features  # noqa: E402

DPI = 200
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR_FALL, COLOR_ADL, COLOR_HARD, COLOR_UNKNOWN = "#eb6834", "#2a78d6", "#1baf7a", "#9a9992"
DATASET_NAMES = {"urfd": "URFD", "le2i": "Le2i"}
MAX_POINTS_PER_CLASS = 1500
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def _save(figure: plt.Figure, out_path: Path) -> Path:
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=DPI)
    plt.close(figure)
    return out_path


def _labeled(meta: pd.DataFrame) -> pd.DataFrame:
    """Chỉ giữ video đã xác định lớp (bỏ 3 video ODD của Le2i)."""
    return meta[meta["has_event_label"] == 1]


def _bar_labels(axis: plt.Axes, positions, bottoms, heights) -> None:
    """Ghi số vào giữa từng đoạn cột, bỏ qua đoạn bằng 0."""
    for position, bottom, height in zip(positions, bottoms, heights):
        if height > 0:
            axis.text(position, bottom + height / 2, str(int(height)), ha="center", va="center", color="white", fontsize=9)


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
    axis.set_title("Số clip theo lớp và nguồn (không tính 3 video chưa xác định của Le2i)", fontsize=10, loc="left")
    axis.legend()
    return _save(figure, out_path)


def plot_duration(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 2: phân bố độ dài clip (giây) theo lớp, mỗi nguồn một ô vì thang thời gian khác nhau."""
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
    """Đếm frame té ngã (trong [fall_start, fall_end], tính cả hai đầu) và bình thường theo nguồn."""
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
    axis.set_xlim(0, 100)
    axis.set_xlabel("Tỷ lệ frame (%)")
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    return _save(figure, out_path)


def plot_scene_distribution(meta: pd.DataFrame, out_path: Path) -> Path:
    """Biểu đồ 4: phân bố bối cảnh/camera; Le2i tách phần 01/02, có cả video chưa xác định."""
    data = meta.copy()
    is_urfd = data["dataset"] == "urfd"
    data["group"] = (data["scene"].str.replace("_", " ").str.capitalize() + " " + data["scene_part"].fillna(""))
    data.loc[is_urfd, "group"] = "URFD " + data.loc[is_urfd, "camera"]
    groups = sorted(data["group"].unique(), key=lambda name: (not name.startswith("URFD"), name))
    figure, axis = plt.subplots(figsize=(7.2, 4))
    bottoms = np.zeros(len(groups))
    for label, name, color in (("fall", "Té ngã", COLOR_FALL), ("adl", "ADL", COLOR_ADL), (None, "Chưa xác định", COLOR_UNKNOWN)):
        mask = data["label"].isna() if label is None else data["label"] == label
        counts = data[mask].groupby("group").size().reindex(groups, fill_value=0).to_numpy()
        axis.bar(groups, counts, bottom=bottoms, color=color, width=0.65, label=name)
        _bar_labels(axis, range(len(groups)), bottoms, counts)
        bottoms = bottoms + counts
    axis.set_ylabel("Số clip")
    axis.grid(axis="x", visible=False)
    axis.legend()
    return _save(figure, out_path)


def _balanced_sample(y: np.ndarray, seed: int) -> np.ndarray:
    """Chọn tối đa MAX_POINTS_PER_CLASS điểm mỗi lớp để t-SNE chạy nhanh và lớp ít không bị che."""
    rng = np.random.default_rng(seed)
    picked = [rng.permutation(np.flatnonzero(y == label))[:MAX_POINTS_PER_CLASS] for label in (0, 1)]
    return np.sort(np.concatenate(picked))


def plot_embedding(X: np.ndarray, y: np.ndarray, feature_source: str, out_path: Path,
                   perplexity: float = 30.0, seed: int = 42) -> Path:
    """Biểu đồ 5: PCA và t-SNE về 2D, tô màu fall/non-fall.

    X: ma trận N×D bất kỳ. y: 0/1. Trên hình ghi rõ nguồn đặc trưng, perplexity và seed.
    """
    chosen = _balanced_sample(y, seed)
    scaled = StandardScaler().fit_transform(X[chosen])
    labels = y[chosen]
    used_perplexity = min(perplexity, (len(chosen) - 1) / 3)
    pca = PCA(n_components=2, random_state=seed)
    pca_points = pca.fit_transform(scaled)
    tsne_points = TSNE(n_components=2, perplexity=used_perplexity, init="pca", random_state=seed).fit_transform(scaled)
    panels = {f"PCA (giải thích {pca.explained_variance_ratio_.sum():.0%} phương sai)": pca_points,
              f"t-SNE (perplexity={used_perplexity:g})": tsne_points}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for axis, (title, points) in zip(axes, panels.items()):
        for label, name, color in ((0, "Không té ngã", COLOR_ADL), (1, "Té ngã", COLOR_FALL)):
            mask = labels == label
            axis.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.65, color=color, linewidths=0,
                         label=f"{name} (n={int(mask.sum())})")
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("Thành phần 1")
        axis.set_ylabel("Thành phần 2")
        axis.legend(fontsize=8)
    figure.suptitle(f"Nguồn đặc trưng: {feature_source} | seed={seed}", fontsize=10, x=0.01, ha="left")
    return _save(figure, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ 5 biểu đồ EDA.")
    parser.add_argument("--meta", required=True, help="metadata.csv")
    parser.add_argument("--out-dir", default="reports/figures", help="thư mục lưu PNG")
    parser.add_argument("--gt-bbox", required=True, help="le2i_gt_bbox.csv")
    parser.add_argument("--tracks", help="tracks.parquet; có thì thay GT làm nguồn bbox của biểu đồ 5")
    parser.add_argument("--config", default="configs/m1.yaml", help="lấy tham số tín hiệu từ đây")
    parser.add_argument("--window-s", type=float, default=1.0, help="độ dài cửa sổ (giây)")
    parser.add_argument("--stride-s", type=float, default=0.25, help="bước trượt (giây)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="perplexity của t-SNE")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    out_dir = Path(args.out_dir)
    written = [plot_clip_counts(meta, out_dir / "eda_1_clip_counts.png"), plot_duration(meta, out_dir / "eda_2_duration.png"),
               plot_frame_balance(meta, out_dir / "eda_3_frame_balance.png"),
               plot_scene_distribution(meta, out_dir / "eda_4_scene_distribution.png")]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gt_boxes = pd.read_csv(args.gt_bbox)
    boxes = pd.read_parquet(args.tracks) if args.tracks else gt_boxes
    X, y, names = build_window_features(boxes, meta, args.window_s, args.stride_s, config["signals"],
                                        config["rules"]["still_speed"], gt_boxes if args.tracks else None)
    source = f"{len(names)} đặc trưng hình học, cửa sổ {args.window_s:g} s, bbox từ {'tracks cache' if args.tracks else 'Le2i GT'}"
    written.append(plot_embedding(X, y, source, out_dir / "eda_5_pca_tsne.png", args.perplexity, args.seed))
    print(f"TÓM TẮT | video trong metadata: {len(meta)} | cửa sổ: {len(y)} (té ngã {int(y.sum())}) | bỏ qua: 0 | Đã ghi: "
          + ", ".join(str(path) for path in written))


if __name__ == "__main__":
    main()
