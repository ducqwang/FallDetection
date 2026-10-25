"""Glue end-to-end: video -> detect/track -> (M1/M2/M3) -> cảnh báo -> EventMetrics.

Kết nối detection_tracking + features + models + evaluate. Dùng cho notebook 06
(đánh giá thống nhất) và cho Web Demo. Phần đọc ảnh/pose cần cv2 + ultralytics.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .detection_tracking import DetectorTracker
from .datasets import make_gt_falls, video_fps, _tracks_by_id
from .evaluate import merge_alarms, evaluate_events, aggregate_over_dataset, EventMetrics
from .features import bbox_geometric_features
from .models.model1_rule import RuleThresholds, predict_sequence

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------- trích crop vùng người theo ID -----------------------
def extract_id_crops(video_source, tracks, size: int = 224):
    """Trả dict[track_id] -> (frames[list[int]], crops np.ndarray[T,3,size,size]).

    Đọc lại video theo frame, cắt vùng bbox của từng ID, resize + chuẩn hóa ImageNet.
    """
    import cv2
    from .preprocessing import read_frames

    by_id = _tracks_by_id(tracks)
    # map frame -> list[(tid, box)]
    frame_boxes: dict[int, list] = {}
    for tid, seq in by_id.items():
        for t in seq:
            frame_boxes.setdefault(t.frame, []).append((tid, t))
    buckets: dict[int, list] = {tid: [] for tid in by_id}
    frames_of: dict[int, list] = {tid: [] for tid in by_id}

    for fi, frame in read_frames(video_source):
        if fi not in frame_boxes:
            continue
        H, W = frame.shape[:2]
        for tid, t in frame_boxes[fi]:
            x1, y1 = max(0, int(t.x)), max(0, int(t.y))
            x2, y2 = min(W, int(t.x + t.w)), min(H, int(t.y + t.h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (size, size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
            buckets[tid].append(crop.transpose(2, 0, 1))
            frames_of[tid].append(fi)
    return {tid: (frames_of[tid], np.stack(c)) for tid, c in buckets.items() if c}


# ----------------------- suy luận cảnh báo cho từng mô hình -----------------------
def alarms_rule(tracks, th: RuleThresholds):
    alarms = []
    for seq in _tracks_by_id(tracks).values():
        alarms += merge_alarms(predict_sequence(seq, th))
    return alarms


def _window_flags(frames, scores, seq_len, stride, thresh=0.5):
    """Từ điểm số cửa sổ -> cờ frame (đánh dấu frame cuối mỗi cửa sổ dương)."""
    flags = [False] * (max(frames) + 1) if frames else []
    for (i0, score) in scores:
        if score >= thresh:
            f = frames[min(i0 + seq_len - 1, len(frames) - 1)]
            flags[f] = True
    return flags


def alarms_sequence_model(id_sequences, predict_fn, seq_len=16, stride=8, thresh=0.5):
    """Chạy mô hình chuỗi (M2/M3) trên từng ID bằng cửa sổ trượt -> cảnh báo.

    id_sequences: dict[tid] -> (frames[list[int]], feats np.ndarray[T, ...]).
    predict_fn: callable(batch np.ndarray[N, seq_len, ...]) -> prob np.ndarray[N].
    """
    alarms = []
    for tid, (frames, feats) in id_sequences.items():
        if len(feats) < seq_len:
            continue
        idxs = list(range(0, len(feats) - seq_len + 1, stride))
        batch = np.stack([feats[i:i + seq_len] for i in idxs])
        probs = predict_fn(batch)
        scores = list(zip(idxs, probs))
        flags = _window_flags(frames, scores, seq_len, stride, thresh)
        alarms += merge_alarms(flags)
    return alarms


# ----------------------- chạy một mô hình trên cả tập test -----------------------
def run_rule_on_testset(items, detector: DetectorTracker, th: RuleThresholds | None = None):
    """Đánh giá M1 (rule-based) trên danh sách VideoItem/dict -> EventMetrics gộp."""
    th = th or RuleThresholds()
    per_video, total_sec = [], 0.0
    for it in items:
        path = it["path"] if isinstance(it, dict) else it.path
        tracks = detector.run(path)
        fps = video_fps(it)
        n_frames = (max((t.frame for t in tracks), default=0) + 1)
        seconds = n_frames / fps
        total_sec += seconds
        gt = make_gt_falls(it, n_frames)
        alarms = alarms_rule(tracks, th)
        per_video.append(evaluate_events(alarms, gt, seconds, fps))
    return aggregate_over_dataset(per_video, total_sec)


def _windows_from_id_seq(frames, feats, gt_falls, seq_len, stride):
    """Cắt cửa sổ trượt + gán nhãn theo chồng lấn với khoảng ngã thật."""
    from .datasets import _label_window
    out = []
    for i in range(0, max(1, len(feats) - seq_len + 1), stride):
        win = feats[i:i + seq_len]
        if len(win) < seq_len:
            continue
        wf = frames[i:i + seq_len]
        out.append((np.asarray(win, np.float32), _label_window(wf, gt_falls)))
    return out


def build_pose_samples(items, seq_len=16, stride=8, pose_model="yolo11n-pose.pt"):
    """Tạo mẫu huấn luyện M2: list[(np.ndarray[seq_len, K*2], label)]."""
    from .features import pose_keypoints
    samples = []
    for it in items:
        path = it["path"] if isinstance(it, dict) else it.path
        kp_by_id = pose_keypoints(path, pose_model)
        fps = video_fps(it)
        n_frames = max((len(v) for v in kp_by_id.values()), default=0)
        gt = make_gt_falls(it, n_frames)
        for _tid, kp in kp_by_id.items():            # kp: [T, K, 2]
            feats = kp.reshape(len(kp), -1)          # [T, K*2]
            frames = list(range(len(feats)))
            samples += _windows_from_id_seq(frames, feats, gt, seq_len, stride)
    return samples


def build_crop_embedding_samples(items, detector: DetectorTracker, backbone="mobilenet_v2",
                                 seq_len=16, stride=8):
    """Tạo mẫu huấn luyện M3 dạng đặc trưng CNN tiền tính (nhẹ RAM/nhanh hội tụ).

    Trả list[(np.ndarray[seq_len, D], label)]. (Dùng CNNGRU với backbone='precomputed'
    hoặc một MLP+GRU; xem notebook 05 để chọn biến thể.)
    """
    from .features import cnn_embeddings
    samples = []
    for it in items:
        path = it["path"] if isinstance(it, dict) else it.path
        tracks = detector.run(path)
        fps = video_fps(it)
        n_frames = (max((t.frame for t in tracks), default=0) + 1)
        gt = make_gt_falls(it, n_frames)
        crops_by_id = extract_id_crops(path, tracks)
        for _tid, (frames, crops) in crops_by_id.items():
            emb = cnn_embeddings(crops, backbone)     # [T, D]
            samples += _windows_from_id_seq(frames, emb, gt, seq_len, stride)
    return samples


def make_predict_fn(model, device=None):
    """Bọc model torch thành predict_fn(batch)->prob (numpy)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def _fn(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(batch, dtype=torch.float32, device=device)
            return torch.sigmoid(model(x)).cpu().numpy()
    return _fn
