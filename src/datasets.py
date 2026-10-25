"""Xây mẫu huấn luyện/đánh giá từ video -> tracks -> chuỗi cửa sổ trượt.

- make_gt_falls: sinh danh sách sự kiện ngã thật (frame) cho một video.
- geometric_windows: cửa sổ đặc trưng hình học (numpy, KHÔNG cần torch) — dùng
  cho rule-based/phân tích và test nhanh.
- PoseSequenceDataset / CropSequenceDataset: Dataset torch cho M2 / M3.

Lưu ý: hàm chạy detector/pose cần cv2 + ultralytics (có trên Kaggle).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .features import bbox_geometric_features


def _field(item, name):
    """Đọc thuộc tính từ VideoItem hoặc dict (splits['_items'])."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def video_fps(item) -> float:
    """fps gốc của video (URFD ~30, Le2i ~25)."""
    return 25.0 if _field(item, "dataset") == "le2i" else 30.0


def make_gt_falls(item, n_frames: int) -> list[tuple[int, int]]:
    """Danh sách (start, end) sự kiện ngã thật cho một video.

    - Le2i: dùng fall_start/fall_end có trong annotation.
    - URFD: không có mốc frame -> coi cả chuỗi fall là một sự kiện ngã
      (chuỗi URFD ngắn, mỗi chuỗi chứa đúng một lần ngã).
    - Video ADL (label 0): không có sự kiện ngã.
    """
    label = _field(item, "label")
    if label == 0:
        return []
    fs = _field(item, "fall_start")
    fe = _field(item, "fall_end")
    if fs is not None and fe is not None:
        return [(int(fs), int(fe))]
    return [(0, max(1, n_frames - 1))]


def _tracks_by_id(tracks):
    """Gom list[Track] theo track_id, sắp theo frame."""
    by_id: dict[int, list] = {}
    for t in tracks:
        by_id.setdefault(t.track_id, []).append(t)
    for k in by_id:
        by_id[k].sort(key=lambda t: t.frame)
    return by_id


def _label_window(frames: list[int], gt_falls: list[tuple[int, int]], tol: int = 0) -> int:
    """Nhãn 1 nếu cửa sổ frame chồng lấn khoảng ngã thật."""
    for s, e in gt_falls:
        if frames[-1] >= s - tol and frames[0] <= e + tol:
            return 1
    return 0


def geometric_windows(tracks, gt_falls, seq_len: int = 16, stride: int = 8):
    """Cửa sổ trượt đặc trưng hình học -> (X[N,5], y[N]). Không cần torch."""
    X, y = [], []
    for _tid, seq in _tracks_by_id(tracks).items():
        for i in range(0, max(1, len(seq) - seq_len + 1), stride):
            win = seq[i:i + seq_len]
            if len(win) < 3:
                continue
            X.append(bbox_geometric_features(win))
            frames = [t.frame for t in win]
            y.append(_label_window(frames, gt_falls))
    if not X:
        return np.zeros((0, 5), np.float32), np.zeros((0,), np.int64)
    return np.stack(X), np.array(y, np.int64)


# ----------------------- Torch datasets (M2, M3) -----------------------
try:
    import torch
    from torch.utils.data import Dataset

    class PoseSequenceDataset(Dataset):
        """Chuỗi keypoint [T, K*2] -> nhãn. samples: list[(np.ndarray[T,K*2], int)]."""
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            x, lab = self.samples[i]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(lab, dtype=torch.float32)

    class CropSequenceDataset(Dataset):
        """Chuỗi ảnh vùng người [T,3,H,W] -> nhãn. samples: list[(np.ndarray[T,3,H,W], int)]."""
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            x, lab = self.samples[i]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(lab, dtype=torch.float32)

    def pad_collate(batch):
        """Gộp batch chuỗi khác độ dài bằng pad về max T."""
        xs, ys = zip(*batch)
        T = max(x.shape[0] for x in xs)
        xs_p = []
        for x in xs:
            if x.shape[0] < T:
                pad = torch.zeros((T - x.shape[0], *x.shape[1:]), dtype=x.dtype)
                x = torch.cat([x, pad], 0)
            xs_p.append(x)
        return torch.stack(xs_p), torch.stack(ys)
except ImportError:
    PoseSequenceDataset = CropSequenceDataset = None

    def pad_collate(batch):  # type: ignore
        raise ImportError("Cần torch cho pad_collate.")
