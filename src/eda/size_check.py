"""Đo xem kích thước khung hình có tách được các cửa sổ đặc trưng không, và ghi cảnh báo vào DATA_CARD.md.

Đầu vào: X (N×D), y (0/1), sizes (N nhãn '320×180'). Đầu ra: dict số đo; khối cảnh báo trong DATA_CARD.md.
Cách đo: silhouette theo nhãn kích thước (−1..1, càng cao càng tách), khoảng cách centroid giữa 2 nhóm
chia cho độ lệch chuẩn gộp, và silhouette riêng trong lớp fall để tách yếu tố lớp ra khỏi kích thước.
Ví dụ: sep = measure_size_separation(X, y, sizes, seed=42); update_data_card("DATA_CARD.md", meta, sep).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..data.frame_size import DEFAULT_WARNING_THRESHOLD, replace_block, size_block

MAX_SILHOUETTE_POINTS = 3000
MIN_POINTS_PER_SIZE = 2


def _silhouette(Z: np.ndarray, groups: np.ndarray, seed: int) -> float | None:
    """Silhouette theo nhóm; None nếu chưa đủ 2 nhóm hoặc nhóm nào dưới MIN_POINTS_PER_SIZE điểm."""
    names, counts = np.unique(groups, return_counts=True)
    if len(names) < 2 or counts.min() < MIN_POINTS_PER_SIZE:
        return None
    sample = min(MAX_SILHOUETTE_POINTS, len(groups))
    return float(silhouette_score(Z, groups, sample_size=sample if sample < len(groups) else None, random_state=seed))


def _centroid_distance(Z: np.ndarray, groups: np.ndarray) -> float:
    """Khoảng cách Euclid giữa 2 centroid lớn nhất chia cho độ lệch chuẩn gộp trong nhóm (0 nếu chỉ có 1 nhóm)."""
    names = sorted(set(groups))
    if len(names) < 2:
        return 0.0
    centroids = np.array([Z[groups == name].mean(axis=0) for name in names])
    spread = np.sqrt(np.mean([Z[groups == name].var(axis=0).sum() for name in names]))
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(centroids) for b in centroids[i + 1:]]
    return float(max(gaps) / max(spread, 1e-9))


def measure_size_separation(X: np.ndarray, y: np.ndarray, sizes: np.ndarray, seed: int = 42) -> dict:
    """Đo độ tách theo kích thước trên đặc trưng đã chuẩn hóa. Khoá: n_windows, sizes, silhouette_size,
    centroid_distance, silhouette_size_fall (None nếu lớp fall chỉ có 1 kích thước), silhouette_label."""
    if not (len(X) == len(y) == len(sizes)):
        raise ValueError(f"X, y, sizes khác độ dài: {len(X)}, {len(y)}, {len(sizes)}")
    Z = StandardScaler().fit_transform(X)
    sizes = np.asarray(sizes)
    fall = y == 1
    silhouette_size = _silhouette(Z, sizes, seed)
    if silhouette_size is None:
        raise ValueError("Cần ít nhất 2 kích thước khung, mỗi kích thước có từ 2 cửa sổ")
    silhouette_label = _silhouette(Z, y, seed)
    return {"n_windows": int(len(y)), "sizes": sorted(set(sizes)), "silhouette_size": silhouette_size,
            "centroid_distance": _centroid_distance(Z, sizes),
            "silhouette_size_fall": _silhouette(Z[fall], sizes[fall], seed),
            "silhouette_label": silhouette_label if silhouette_label is not None else float("nan")}


def update_data_card(path: str | Path, meta: pd.DataFrame, separation: dict | None,
                     threshold: float = DEFAULT_WARNING_THRESHOLD) -> Path:
    """Ghi (hoặc ghi đè) khối kích thước khung trong DATA_CARD.md; phần còn lại của file giữ nguyên."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else "# DATA CARD — nhom05-falldet-processed\n"
    path.write_text(replace_block(text, size_block(meta, separation, threshold)), encoding="utf-8")
    return path
