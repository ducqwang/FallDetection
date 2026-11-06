"""Biểu đồ 5: PCA và t-SNE về 2D, tô theo lớp hoặc theo kích thước khung.

Đầu vào: ma trận đặc trưng N×D bất kỳ, nhãn 0/1, (tuỳ chọn) kích thước khung của từng điểm.
Đầu ra: PNG dpi 200. Ví dụ: plot_embedding(X, y, "9 đặc trưng", Path("a.png"), color_by="size", sizes=sizes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .style import COLOR_ADL, COLOR_FALL, SIZE_COLORS, _save, plt

MAX_POINTS_PER_CLASS = 1500


def _balanced_sample(y: np.ndarray, seed: int) -> np.ndarray:
    """Chọn tối đa MAX_POINTS_PER_CLASS điểm mỗi lớp để t-SNE chạy nhanh và lớp ít không bị che."""
    rng = np.random.default_rng(seed)
    picked = [rng.permutation(np.flatnonzero(y == label))[:MAX_POINTS_PER_CLASS] for label in (0, 1)]
    return np.sort(np.concatenate(picked))


def _embed(X: np.ndarray, y: np.ndarray, seed: int, perplexity: float, sizes: np.ndarray | None) -> dict:
    """Chuẩn hóa, lấy mẫu cân bằng lớp rồi giảm về 2D bằng PCA và t-SNE (chỉ tính 1 lần cho cả 2 bản tô màu)."""
    if sizes is not None and len(sizes) != len(y):
        raise ValueError(f"sizes có {len(sizes)} phần tử nhưng y có {len(y)}")
    chosen = _balanced_sample(y, seed)
    scaled = StandardScaler().fit_transform(X[chosen])
    used_perplexity = min(perplexity, (len(chosen) - 1) / 3)
    pca = PCA(n_components=2, random_state=seed)
    pca_points = pca.fit_transform(scaled)
    tsne_points = TSNE(n_components=2, perplexity=used_perplexity, init="pca", random_state=seed).fit_transform(scaled)
    return {"labels": y[chosen], "sizes": None if sizes is None else np.asarray(sizes)[chosen],
            "panels": {f"PCA (giải thích {pca.explained_variance_ratio_.sum():.0%} phương sai)": pca_points,
                       f"t-SNE (perplexity={used_perplexity:g})": tsne_points}}


def _color_groups(color_by: str, labels: np.ndarray, sizes: np.ndarray | None) -> list[tuple[str, str, np.ndarray]]:
    """Danh sách (tên, màu, mask) cho color_by = 'label' (fall/ADL) hoặc 'size' (kích thước khung)."""
    if color_by == "label":
        return [(name, color, labels == value) for value, name, color in ((0, "Không té ngã", COLOR_ADL), (1, "Té ngã", COLOR_FALL))]
    if color_by == "size":
        if sizes is None:
            raise ValueError("color_by='size' cần tham số sizes (kích thước khung của từng điểm)")
        return [(str(size), SIZE_COLORS[index % len(SIZE_COLORS)], sizes == size) for index, size in enumerate(sorted(set(sizes)))]
    raise ValueError(f"color_by phải là 'label' hoặc 'size', nhận {color_by!r}")


def _draw_embedding(embedded: dict, color_by: str, feature_source: str, out_path: Path, seed: int) -> Path:
    groups = _color_groups(color_by, embedded["labels"], embedded["sizes"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for axis, (title, points) in zip(axes, embedded["panels"].items()):
        for name, color, mask in groups:
            axis.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.65, color=color, linewidths=0,
                         label=f"{name} (n={int(mask.sum())})")
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_xlabel("Thành phần 1")
        axis.set_ylabel("Thành phần 2")
        axis.legend(fontsize=8)
    tint = "tô theo lớp" if color_by == "label" else "tô theo kích thước khung"
    figure.suptitle(f"Nguồn đặc trưng: {feature_source} | seed={seed} | {tint}", fontsize=10, x=0.01, ha="left")
    return _save(figure, out_path)


def plot_embedding(X: np.ndarray, y: np.ndarray, feature_source: str, out_path: Path,
                   perplexity: float = 30.0, seed: int = 42, color_by: str = "label",
                   sizes: np.ndarray | None = None) -> Path:
    """Biểu đồ 5: PCA và t-SNE về 2D. color_by='label' tô fall/non-fall, 'size' tô theo kích thước khung.

    X: ma trận N×D bất kỳ. y: 0/1. sizes: mảng N nhãn kích thước ('320×180'), bắt buộc khi color_by='size'.
    Trên hình ghi rõ nguồn đặc trưng, perplexity và seed.
    """
    if color_by == "size" and sizes is None:
        raise ValueError("color_by='size' cần tham số sizes (kích thước khung của từng điểm)")
    return _draw_embedding(_embed(X, y, seed, perplexity, sizes), color_by, feature_source, out_path, seed)


def plot_embedding_pair(X: np.ndarray, y: np.ndarray, sizes: np.ndarray, feature_source: str, out_dir: Path,
                        perplexity: float = 30.0, seed: int = 42) -> list[Path]:
    """Vẽ 2 bản của biểu đồ 5 từ cùng 1 phép giảm chiều: tô theo lớp và tô theo kích thước khung."""
    embedded = _embed(X, y, seed, perplexity, sizes)
    return [_draw_embedding(embedded, "label", feature_source, out_dir / "eda_5_pca_tsne.png", seed),
            _draw_embedding(embedded, "size", feature_source, out_dir / "eda_5b_pca_tsne_by_size.png", seed)]
