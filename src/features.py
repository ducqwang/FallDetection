"""Trích đặc trưng + giảm chiều + trực quan hóa (đáp ứng §2.2 đề cương).

- Đặc trưng hình học từ khung bao (cho M1 và phân tích).
- Đặc trưng keypoint pose (cho M2).
- Embedding CNN đóng băng (cho M3).
- PCA / t-SNE để giảm chiều + vẽ scatter tô màu theo nhãn (BẮT BUỘC).
"""
from __future__ import annotations
import numpy as np


def bbox_geometric_features(track_seq) -> np.ndarray:
    """Từ chuỗi khung bao (list[Track]) -> đặc trưng hình học theo thời gian.

    Trả vector gồm: tỷ lệ w/h, vận tốc tâm theo phương dọc, gia tốc, mức bất động.
    Dùng cho Mô hình 1 (rule-based) và cho scatter PCA/t-SNE.
    """
    if len(track_seq) < 2:
        return np.zeros(5, dtype=np.float32)
    cx = np.array([t.x + t.w / 2 for t in track_seq])
    cy = np.array([t.y + t.h / 2 for t in track_seq])
    ar = np.array([t.w / max(t.h, 1e-6) for t in track_seq])     # aspect ratio
    vy = np.diff(cy)                                             # vận tốc dọc
    ay = np.diff(vy) if len(vy) > 1 else np.array([0.0])          # gia tốc dọc
    speed = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    stillness = float((speed[-3:] < speed.mean() * 0.3).mean()) if len(speed) >= 3 else 0.0
    return np.array([ar.max(), ar[-1], vy.max(), ay.max(), stillness], dtype=np.float32)


def normalize_keypoints(kp_xy: np.ndarray, box) -> np.ndarray:
    """Chuẩn hóa keypoint về [0,1] theo khung bao -> bất biến vị trí & kích thước.

    kp_xy: [K,2] toạ độ pixel; box: (x, y, w, h). Trả [K,2].
    """
    x, y, w, h = box
    out = kp_xy.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - x) / max(w, 1e-6)
    out[:, 1] = (out[:, 1] - y) / max(h, 1e-6)
    return np.clip(out, -0.5, 1.5)


def pose_keypoints(video_source, model_name: str = "yolo11n-pose.pt",
                   conf: float = 0.25, tracker: str = "bytetrack.yaml"):
    """Trích keypoint theo ID người bằng YOLO11-pose (có tracking).

    Trả về dict[track_id] -> np.ndarray[T, K, 2] đã chuẩn hóa theo khung bao.
    """
    from ultralytics import YOLO
    model = YOLO(model_name)
    by_id: dict[int, list] = {}
    results = model.track(source=video_source, persist=True, tracker=tracker,
                          classes=[0], conf=conf, stream=True, verbose=False)
    for r in results:
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue
        ids = r.boxes.id.int().cpu().numpy()
        boxes = r.boxes.xywh.cpu().numpy()
        kps = r.keypoints.xy.cpu().numpy()          # [n, K, 2]
        for tid, box, kp in zip(ids, boxes, kps):
            cx, cy, w, h = box
            norm = normalize_keypoints(kp, (cx - w / 2, cy - h / 2, w, h))
            by_id.setdefault(int(tid), []).append(norm)
    return {k: np.stack(v) for k, v in by_id.items() if len(v) > 0}


def _build_backbone(backbone: str):
    import torch
    from torchvision import models
    if backbone == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        feat = torch.nn.Sequential(net.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = torch.nn.Sequential(*list(net.children())[:-1], torch.nn.Flatten())
    feat.eval()
    for p in feat.parameters():
        p.requires_grad = False
    return feat


def cnn_embeddings(crops, backbone: str = "mobilenet_v2", device: str | None = None):
    """Backbone CNN pretrained ĐÓNG BĂNG -> đặc trưng mỗi frame.

    crops: np.ndarray[T,3,H,W] đã chuẩn hóa ImageNet. Trả np.ndarray[T, D].
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feat = _build_backbone(backbone).to(device)
    x = torch.as_tensor(np.asarray(crops), dtype=torch.float32, device=device)
    with torch.no_grad():
        emb = feat(x)
    return emb.cpu().numpy()


def reduce_and_plot(X: np.ndarray, y: np.ndarray, method: str = "tsne", out_png: str = "outputs/embedding.png"):
    """Giảm chiều (PCA/t-SNE) và vẽ scatter 2D tô màu theo nhãn fall/non-fall."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == "pca":
        Z = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X) // 4))).fit_transform(X)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for lab, name, c in [(0, "Bình thường", "#1f77b4"), (1, "Té ngã", "#d62728")]:
        m = y == lab
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6, label=name, color=c)
    plt.legend(); plt.title(f"{method.upper()} đặc trưng (fall vs non-fall)")
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return out_png
