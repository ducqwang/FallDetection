"""Mô hình 3 — CNN (đóng băng) + GRU trên ảnh vùng người.

Backbone pretrained (MobileNetV2/ResNet) đóng băng -> đặc trưng mỗi frame -> GRU.
Chống overfit dữ liệu nhỏ: đóng băng backbone, augmentation mạnh, dropout.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torchvision import models

    class CNNGRU(nn.Module):
        def __init__(self, backbone: str = "mobilenet_v2", hidden: int = 128,
                     dropout: float = 0.4):
            super().__init__()
            if backbone == "mobilenet_v2":
                net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                self.feat = net.features
                feat_dim = 1280
            else:
                net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.feat = nn.Sequential(*list(net.children())[:-1])
                feat_dim = 512
            for p in self.feat.parameters():      # ĐÓNG BĂNG backbone
                p.requires_grad = False
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):                      # x: [B, T, 3, H, W]
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            with torch.no_grad():
                f = self.pool(self.feat(x)).flatten(1)   # [B*T, D]
            f = f.view(B, T, -1)
            out, _ = self.gru(f)
            return self.head(out[:, -1]).squeeze(-1)

    class SeqGRU(nn.Module):
        """GRU trên đặc trưng CNN TIỀN TÍNH [B, T, D] (backbone đóng băng, tính sẵn).

        Tương đương toán học với CNNGRU (backbone frozen) nhưng nhanh hơn nhiều trên
        Kaggle vì chỉ chạy CNN một lần. Dùng với pipeline.build_crop_embedding_samples.
        """
        def __init__(self, in_dim: int = 1280, hidden: int = 128, dropout: float = 0.4):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):                      # x: [B, T, D]
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)
except ImportError:
    CNNGRU = SeqGRU = None


def train(model, loader_train, loader_val, cfg, ckpt_path: str = "outputs/model3_best.pt"):
    """Huấn luyện M3 (chỉ cập nhật GRU+head; backbone đã đóng băng). Lưu best."""
    from ..trainer import fit
    return fit(model, loader_train, loader_val, cfg, ckpt_path)
