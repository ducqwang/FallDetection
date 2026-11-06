"""Giao diện chung của biểu đồ EDA: bảng màu, rcParams, lưu hình, ghi số lên cột.

Dùng chung cho plots.py và embedding.py để 2 file không import vòng.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 200
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR_FALL, COLOR_ADL, COLOR_HARD = "#eb6834", "#2a78d6", "#1baf7a"
SIZE_COLORS = ["#7b5ea7", "#e0a100", "#1baf7a", "#52514e"]
DATASET_NAMES = {"urfd": "URFD", "le2i": "Le2i"}
MIN_INSIDE_LABEL = 4  # đoạn cột thấp hơn số clip này không đủ chỗ ghi số bên trong
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


def _bar_labels(axis: plt.Axes, positions, bottoms, heights) -> None:
    """Ghi số vào giữa từng đoạn cột; đoạn quá mỏng ghi ngay trên đoạn, bỏ qua đoạn bằng 0."""
    for position, bottom, height in zip(positions, bottoms, heights):
        if height >= MIN_INSIDE_LABEL:
            axis.text(position, bottom + height / 2, str(int(height)), ha="center", va="center", color="white", fontsize=9)
        elif height > 0:
            axis.text(position, bottom + height, str(int(height)), ha="center", va="bottom", color=INK, fontsize=9)
