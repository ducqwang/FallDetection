"""Đọc video, trích frame (lấy mẫu theo fps) và tăng cường dữ liệu."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator

import numpy as np


def read_frames(video_path: str | Path, fps_sample: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Duyệt frame của video. URFD là chuỗi PNG, Le2i là .avi -> xử lý cả hai.

    Yields (frame_index, frame_bgr).
    """
    import cv2
    p = Path(video_path)
    if p.is_dir():  # URFD: thư mục chứa PNG
        frames = sorted(p.glob("*.png"))
        step = 1
        if fps_sample:  # URFD ~30fps
            step = max(1, round(30 / fps_sample))
        for i, fp in enumerate(frames):
            if i % step == 0:
                yield i, cv2.imread(str(fp))
    else:  # Le2i: file video
        cap = cv2.VideoCapture(str(p))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(src_fps / fps_sample)) if fps_sample else 1
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
        cap.release()


def build_augment(train: bool = True):
    """Augmentation nhẹ cho dữ liệu nhỏ: lật ngang, đổi sáng, mờ, cắt xén.

    Trả về callable(img)->img. Dùng albumentations nếu có, fallback OpenCV.
    """
    try:
        import albumentations as A
        if not train:
            return A.Compose([A.Resize(224, 224)])
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=0.5),
            A.Resize(224, 224),
        ])
    except ImportError:
        import cv2
        def _aug(img):
            return cv2.resize(img, (224, 224))
        return _aug


# Gợi ý augmentation theo thời gian (video-level): đổi tốc độ phát (±), bỏ frame.
# TODO (TV1): hàm time_augment(sequence) cho M2/M3.
