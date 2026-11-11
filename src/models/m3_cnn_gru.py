"""M3: CNN (backbone đóng băng) + GRU — bản TÍCH HỢP, dùng chung hạ tầng của nhóm.

Khác với notebooks/05 cũ (tự index dữ liệu, tự chạy YOLO, tự đánh giá), bản này tái dùng
đúng hợp đồng chung của TV1/TV2:
  - Dữ liệu  : src.data.loader (metadata.csv, splits.json, read_frames) — KHÔNG tự quét thư mục.
  - Nhãn     : src.data.labels.window_label + select_main_track — KHÔNG tự định nghĩa sự kiện.
  - Tracks   : tracks.parquet (chính thức của TV1: scripts/make_tracks.py) — KHÔNG tự chạy detector.
  - Đánh giá : scripts/evaluate.py trên predictions.csv — CÙNG độ đo/T/split với M1, M2.

Việc riêng của M3: crop người theo bbox -> nhúng CNN đóng băng -> GRU -> fall_score mỗi frame.

CLI (chạy sau khi đã có tracks.parquet):
  Huấn luyện:
    python -m src.models.m3_cnn_gru train --tracks tracks.parquet --meta metadata.csv \\
        --root <data_root> --dataset le2i --ckpt outputs/m3_best.pt
  Dự đoán (ghi predictions.csv theo schema chung):
    python -m src.models.m3_cnn_gru predict --tracks tracks.parquet --meta metadata.csv \\
        --root <data_root> --ckpt outputs/m3_best.pt --out predictions.csv
Sau đó: python scripts/evaluate.py --pred predictions.csv --meta metadata.csv --split test --dataset le2i
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.common import METADATA_DTYPES, PREDICTION_COLUMNS
from ..data.labels import select_main_track, window_label
from ..data.loader import data_root, get_split, read_frames

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

DEFAULTS = {
    "backbone": "mobilenet_v2",
    "img_size": 224,
    "seq_len": 16,
    "stride": 8,
    "frame_stride": 2,     # nhúng 1 frame mỗi 2 frame gốc để nhẹ GPU (Le2i ~25fps)
    "hidden": 128,
    "dropout": 0.4,
    "lr": 1e-3,
    "epochs": 30,
    "patience": 6,
    "pos_weight": 3.0,
    "embed_batch": 64,
}
FEATURE_DIM = {"mobilenet_v2": 1280, "resnet18": 512}


# ----------------------------------------------------------------------------
# Mô hình: backbone đóng băng + GRU trên đặc trưng tiền tính
# ----------------------------------------------------------------------------
def build_backbone(backbone: str):
    """Trả về (module trích đặc trưng -> vector [B, D], D). Backbone pretrained, ĐÓNG BĂNG."""
    import torch.nn as nn
    from torchvision import models

    if backbone == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        feat = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        dim = FEATURE_DIM[backbone]
    elif backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        dim = FEATURE_DIM[backbone]
    else:
        raise ValueError(f"backbone không hỗ trợ: {backbone}")
    for p in feat.parameters():
        p.requires_grad = False
    feat.eval()
    return feat, dim


def build_gru(in_dim: int, hidden: int = 128, dropout: float = 0.4):
    """GRU phân loại chuỗi đặc trưng [B, T, in_dim] -> logit [B] (chưa qua sigmoid)."""
    import torch.nn as nn

    class SeqGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):            # x: [B, T, in_dim]
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    return SeqGRU()


# ----------------------------------------------------------------------------
# Trích đặc trưng CNN cho một track (crop theo bbox -> nhúng)
# ----------------------------------------------------------------------------
def _preprocess_crop(frame_bgr: np.ndarray, box, size: int) -> np.ndarray | None:
    """Cắt vùng bbox (x1,y1,x2,y2 pixel) -> resize size×size -> RGB chuẩn ImageNet CHW float32."""
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = cv2.resize(frame_bgr[y1:y2, x1:x2], (size, size))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def embed_track(frames: np.ndarray, track_df: pd.DataFrame, backbone_model, device: str,
                size: int, frame_stride: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Nhúng CNN cho 1 track -> (frame_idx đã nhúng [N], đặc trưng [N, D]).

    frames: T×H×W×3 BGR của cả video (index theo frame_idx 0-based).
    track_df: các dòng track của đúng (video, track_id), có frame_idx + x1..y2.
    """
    import torch

    rows = track_df.sort_values("frame_idx")
    if frame_stride > 1:
        rows = rows.iloc[::frame_stride]
    kept_idx: list[int] = []
    crops: list[np.ndarray] = []
    for r in rows.itertuples():
        fi = int(r.frame_idx)
        if fi < 0 or fi >= len(frames):
            continue
        chw = _preprocess_crop(frames[fi], (r.x1, r.y1, r.x2, r.y2), size)
        if chw is None:
            continue
        kept_idx.append(fi)
        crops.append(chw)
    if not crops:
        return np.empty(0, np.int64), np.empty((0, 0), np.float32)
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = np.stack(crops[start:start + batch])
            t = torch.from_numpy(chunk).to(device)
            embs.append(backbone_model(t).cpu().numpy())
    return np.asarray(kept_idx, np.int64), np.concatenate(embs).astype(np.float32)


def make_windows(frame_idx: np.ndarray, emb: np.ndarray, seq_len: int, stride: int):
    """Cắt cửa sổ trượt trên chuỗi đã nhúng. Yield (start_frame, end_frame, emb_window [seq_len, D]).

    Chuỗi ngắn hơn seq_len -> một cửa sổ, đệm bằng lặp lại frame đầu (giữ cú ngã ngắn của Le2i).
    """
    n, d = emb.shape
    if n == 0:
        return
    if n < seq_len:
        pad = np.repeat(emb[:1], seq_len - n, axis=0)
        yield int(frame_idx[0]), int(frame_idx[-1]), np.concatenate([pad, emb])
        return
    for i in range(0, n - seq_len + 1, stride):
        yield int(frame_idx[i]), int(frame_idx[i + seq_len - 1]), emb[i:i + seq_len]


# ----------------------------------------------------------------------------
# Dựng mẫu huấn luyện từ hạ tầng chung
# ----------------------------------------------------------------------------
def _gt_bbox_of(gt_bbox: pd.DataFrame | None, video_id: str) -> pd.DataFrame | None:
    if gt_bbox is None:
        return None
    return gt_bbox[gt_bbox["video_id"] == video_id]


def build_windows_for_ids(video_ids, meta: pd.DataFrame, tracks: pd.DataFrame, root,
                          backbone_model, device: str, cfg: dict,
                          gt_bbox: pd.DataFrame | None, labeled: bool):
    """Duyệt các video_id, nhúng track chính, cắt cửa sổ. Trả (X [M, T, D], y [M] hoặc None).

    labeled=True: gán nhãn cửa sổ bằng window_label (dùng cho train/val).
    """
    meta_by_id = meta.set_index("video_id")
    X: list[np.ndarray] = []
    y: list[int] = []
    for vid in video_ids:
        if vid not in meta_by_id.index:
            continue
        row = meta_by_id.loc[vid]
        vt = tracks[tracks["video_id"] == vid]
        if vt.empty:
            continue
        main = select_main_track(vt, {**row.to_dict(), "video_id": vid}, _gt_bbox_of(gt_bbox, vid))
        if main is None:
            continue
        tdf = vt[vt["track_id"] == main]
        frames = read_frames(vid, root=root)
        fidx, emb = embed_track(frames, tdf, backbone_model, device,
                                cfg["img_size"], cfg["frame_stride"], cfg["embed_batch"])
        if emb.shape[0] == 0:
            continue
        for start_f, end_f, window in make_windows(fidx, emb, cfg["seq_len"], cfg["stride"]):
            X.append(window)
            if labeled:
                y.append(window_label({**row.to_dict(), "video_id": vid}, start_f, end_f))
    if not X:
        return np.empty((0, cfg["seq_len"], 0), np.float32), (np.empty(0, np.int64) if labeled else None)
    return np.stack(X).astype(np.float32), (np.asarray(y, np.int64) if labeled else None)


# ----------------------------------------------------------------------------
# Huấn luyện
# ----------------------------------------------------------------------------
def _f1(logits, targets, thr: float = 0.5) -> float:
    import torch

    pred = (torch.sigmoid(logits) >= thr).int()
    tgt = targets.int()
    tp = int(((pred == 1) & (tgt == 1)).sum())
    fp = int(((pred == 1) & (tgt == 0)).sum())
    fn = int(((pred == 0) & (tgt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def fit(model, x_tr, y_tr, x_va, y_va, cfg: dict, ckpt_path: str, device: str) -> float:
    """Huấn luyện GRU (backbone đã đóng băng), tối ưu F1 cửa sổ trên val, lưu best. Trả best F1."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]], device=device))
    tr = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32))),
                    batch_size=32, shuffle=True)
    xv = torch.from_numpy(x_va).to(device)
    yv = torch.from_numpy(y_va.astype(np.float32)).to(device)
    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f1 = _f1(model(xv), yv) if len(yv) else 0.0
        print(f"  epoch {epoch:02d} | val_f1 = {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"  early-stop tại epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": x_tr.shape[-1],
                "hidden": cfg["hidden"], "dropout": cfg["dropout"], "backbone": cfg["backbone"]}, ckpt_path)
    return best_f1


def train_cli(args) -> None:
    """Huấn luyện M3 trên tập train/val của một bộ dữ liệu (within-dataset)."""
    import torch

    from ..utils import set_seed
    set_seed(args.seed)
    cfg = {**DEFAULTS, "backbone": args.backbone, "epochs": args.epochs}
    root = data_root(args.root)
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    tracks = pd.read_parquet(args.tracks)
    gt_bbox = pd.read_csv(args.gt_bbox) if args.gt_bbox else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone_model, _ = build_backbone(cfg["backbone"])
    backbone_model = backbone_model.to(device)

    train_ids = get_split(args.dataset, "train", root)
    val_ids = get_split(args.dataset, "val", root)
    print(f"train {len(train_ids)} video | val {len(val_ids)} video | device {device}")
    x_tr, y_tr = build_windows_for_ids(train_ids, meta, tracks, root, backbone_model, device, cfg, gt_bbox, True)
    x_va, y_va = build_windows_for_ids(val_ids, meta, tracks, root, backbone_model, device, cfg, gt_bbox, True)
    print(f"cửa sổ train {len(x_tr)} (dương {int(y_tr.sum())}) | val {len(x_va)} (dương {int(y_va.sum())})")
    if len(x_tr) == 0:
        raise SystemExit("Không dựng được cửa sổ train nào — kiểm tra tracks/metadata/splits.")
    model = build_gru(x_tr.shape[-1], cfg["hidden"], cfg["dropout"])
    best = fit(model, x_tr, y_tr, x_va, y_va, cfg, args.ckpt, device)
    print(f"TÓM TẮT | best val F1 = {best:.3f} | Đã lưu: {args.ckpt}")


# ----------------------------------------------------------------------------
# Dự đoán -> predictions.csv (schema chung)
# ----------------------------------------------------------------------------
def predict(meta: pd.DataFrame, tracks: pd.DataFrame, root, model, backbone_model,
            device: str, cfg: dict) -> pd.DataFrame:
    """Chấm fall_score mỗi (video_id, frame_idx, track_id) cho MỌI track. Trả bảng predictions."""
    import torch

    meta_ids = set(meta["video_id"])
    tables: list[pd.DataFrame] = []
    model.eval()
    for vid, vt in tracks.groupby("video_id"):
        if vid not in meta_ids:
            continue
        frames = read_frames(vid, root=root)
        for tid, tdf in vt.groupby("track_id"):
            fidx, emb = embed_track(frames, tdf, backbone_model, device,
                                    cfg["img_size"], cfg["frame_stride"], cfg["embed_batch"])
            if emb.shape[0] == 0:
                continue
            pos = {int(f): i for i, f in enumerate(fidx)}
            score = np.zeros(len(fidx), np.float32)
            with torch.no_grad():
                for start_f, end_f, window in make_windows(fidx, emb, cfg["seq_len"], cfg["stride"]):
                    prob = float(torch.sigmoid(model(torch.from_numpy(window[None]).to(device)))[0])
                    lo, hi = pos[start_f], pos[end_f]
                    score[lo:hi + 1] = np.maximum(score[lo:hi + 1], prob)
            tables.append(pd.DataFrame({"video_id": vid, "frame_idx": fidx,
                                        "track_id": int(tid), "fall_score": score}))
    if not tables:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(tables, ignore_index=True)[PREDICTION_COLUMNS]


def predict_cli(args) -> None:
    """Nạp checkpoint, chấm điểm toàn bộ tracks và ghi predictions.csv."""
    import torch

    root = data_root(args.root)
    meta = pd.read_csv(args.meta, dtype=METADATA_DTYPES)
    tracks = pd.read_parquet(args.tracks)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = {**DEFAULTS, "backbone": ckpt.get("backbone", DEFAULTS["backbone"])}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone_model, _ = build_backbone(cfg["backbone"])
    backbone_model = backbone_model.to(device)
    model = build_gru(ckpt["in_dim"], ckpt["hidden"], ckpt.get("dropout", cfg["dropout"]))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    predictions = predict(meta, tracks, root, model, backbone_model, device, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(f"TÓM TẮT | video có dự đoán: {predictions['video_id'].nunique()} | dòng: {len(predictions)} "
          f"| Đã ghi: {args.out}")


def main() -> None:
    """CLI: hai lệnh con 'train' và 'predict'."""
    parser = argparse.ArgumentParser(description="M3 CNN+GRU (bản tích hợp, dùng chung hạ tầng nhóm).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="huấn luyện trên train/val của 1 bộ")
    pt.add_argument("--tracks", required=True, help="tracks.parquet")
    pt.add_argument("--meta", required=True, help="metadata.csv")
    pt.add_argument("--root", help="thư mục dữ liệu (mặc định: tự dò theo loader.data_root)")
    pt.add_argument("--dataset", required=True, choices=["urfd", "le2i"])
    pt.add_argument("--gt-bbox", help="annotations/le2i_gt_bbox.csv (giúp chọn track chính cho Le2i)")
    pt.add_argument("--backbone", default=DEFAULTS["backbone"], choices=["mobilenet_v2", "resnet18"])
    pt.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--ckpt", required=True, help="đường dẫn lưu checkpoint")
    pt.set_defaults(func=train_cli)

    pp = sub.add_parser("predict", help="chấm điểm toàn bộ tracks -> predictions.csv")
    pp.add_argument("--tracks", required=True, help="tracks.parquet")
    pp.add_argument("--meta", required=True, help="metadata.csv")
    pp.add_argument("--root", help="thư mục dữ liệu (mặc định: tự dò)")
    pp.add_argument("--ckpt", required=True, help="checkpoint đã huấn luyện")
    pp.add_argument("--out", required=True, help="predictions.csv sẽ ghi")
    pp.set_defaults(func=predict_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
