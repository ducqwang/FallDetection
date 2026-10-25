"""Vòng huấn luyện dùng chung cho M2 (Pose+LSTM) và M3 (CNN+GRU).

Tối ưu F1 mức cửa sổ trên tập val (proxy cho F1 sự kiện), BCEWithLogitsLoss,
Adam chỉ cập nhật tham số requires_grad=True (backbone M3 đã đóng băng),
early-stopping + lưu checkpoint tốt nhất. Cố định seed trước khi gọi.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np


def _f1_from_logits(logits, targets, thresh: float = 0.5):
    import torch
    prob = torch.sigmoid(logits)
    pred = (prob >= thresh).int()
    tgt = targets.int()
    tp = int(((pred == 1) & (tgt == 1)).sum())
    fp = int(((pred == 1) & (tgt == 0)).sum())
    fn = int(((pred == 0) & (tgt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def fit(model, loader_train, loader_val, cfg: dict, ckpt_path: str,
        device: str | None = None):
    """Huấn luyện và lưu best checkpoint. Trả về (best_f1, lịch sử).

    cfg cần: lr, epochs; (tùy chọn) patience, pos_weight.
    """
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.get("lr", 1e-3))
    pos_w = torch.tensor([cfg.get("pos_weight", 3.0)], device=device)   # lớp té ngã hiếm
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    epochs = cfg.get("epochs", 30)
    patience = cfg.get("patience", 8)

    best_f1, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(yb)
        tr_loss /= max(1, len(loader_train.dataset))

        # validate
        model.eval()
        logits_all, tgts_all = [], []
        with torch.no_grad():
            for xb, yb in loader_val:
                logits_all.append(model(xb.to(device)).cpu())
                tgts_all.append(yb)
        if logits_all:
            import torch as T
            f1 = _f1_from_logits(T.cat(logits_all), T.cat(tgts_all))
        else:
            f1 = 0.0
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_f1": f1})
        print(f"[ep {ep:02d}] train_loss={tr_loss:.4f}  val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1, best_state, wait = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop tại epoch {ep} (best val_f1={best_f1:.4f}).")
                break

    if best_state is not None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        import torch as T
        T.save(best_state, ckpt_path)
        model.load_state_dict(best_state)
        print(f"Đã lưu best checkpoint -> {ckpt_path} (val_f1={best_f1:.4f})")
    return best_f1, hist
