"""Mô hình 2 — Skeleton + chuỗi thời gian (keypoint -> LSTM/GRU).

Đầu vào: chuỗi keypoint đã chuẩn hóa [T, K*2]. Đầu ra: xác suất té ngã.
Nhẹ, ít tham số, phù hợp dữ liệu nhỏ.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    class PoseLSTM(nn.Module):
        def __init__(self, in_dim: int = 34, hidden: int = 128, layers: int = 1,
                     dropout: float = 0.3, rnn: str = "lstm"):
            super().__init__()
            RNN = nn.LSTM if rnn == "lstm" else nn.GRU
            self.rnn = RNN(in_dim, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):            # x: [B, T, in_dim]
            out, _ = self.rnn(x)
            return self.head(out[:, -1]).squeeze(-1)   # logit [B]
except ImportError:
    PoseLSTM = None  # torch chưa cài (ví dụ môi trường chỉ dựng kế hoạch)


def train(model, loader_train, loader_val, cfg, ckpt_path: str = "outputs/model2_best.pt"):
    """Huấn luyện M2 (ủy quyền cho trainer.fit). Lưu best -> ckpt_path."""
    from ..trainer import fit
    return fit(model, loader_train, loader_val, cfg, ckpt_path)
