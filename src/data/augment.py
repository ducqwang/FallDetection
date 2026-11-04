"""Việc 3: tăng cường clip khi train, không lưu ra đĩa.

Đầu vào: clip ndarray T×H×W×3 uint8 (BGR). Đầu ra: clip mới cùng H×W, T có thể đổi khi đổi tốc độ.
Một lần gọi dùng cùng tham số cho mọi frame của clip. Mọi tham số nằm trong DEFAULT_AUG_CONFIG.
Ví dụ: clip2, applied = augment_clip(clip, seed=42); save_augment_preview(clip, "reports/figures/augment_demo.png").
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_AUG_CONFIG: dict = {
    "hflip": {"p": 0.5},
    "brightness_contrast": {"p": 0.5, "brightness": 0.2, "contrast": 0.2},
    "blur": {"p": 0.2, "kernel_sizes": [3, 5]},
    "crop": {"p": 0.5, "min_scale": 0.9},
    "speed": {"p": 0.3, "min_factor": 0.75, "max_factor": 1.25},
}
# Tham số cố định cho hình minh họa trước/sau trong báo cáo.
PREVIEW_PARAMS: dict = {
    "lật ngang": {},
    "đổi sáng/tương phản": {"brightness": 0.2, "contrast": 0.2},
    "làm mờ": {"kernel_size": 5},
    "cắt xén nhẹ": {"scale": 0.9, "x_ratio": 0.5, "y_ratio": 0.5},
    "tăng tốc 1.25 lần": {"factor": 1.25},
}
PIXEL_MAX = 255


def horizontal_flip(clip: np.ndarray) -> np.ndarray:
    """Lật ngang mọi frame."""
    return clip[:, :, ::-1].copy()


def brightness_contrast(clip: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Đổi sáng và tương phản: pixel * (1 + contrast) + brightness * 255."""
    adjusted = clip.astype(np.float32) * (1.0 + contrast) + brightness * PIXEL_MAX
    return np.clip(adjusted, 0, PIXEL_MAX).astype(np.uint8)


def blur(clip: np.ndarray, kernel_size: int) -> np.ndarray:
    """Làm mờ Gaussian. kernel_size là số lẻ, đơn vị pixel."""
    return np.stack([cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0) for frame in clip])


def crop_resize(clip: np.ndarray, scale: float, x_ratio: float, y_ratio: float) -> np.ndarray:
    """Cắt vùng bằng `scale` kích thước gốc rồi phóng về kích thước cũ.

    x_ratio, y_ratio ∈ [0, 1] là vị trí vùng cắt trong phần dư.
    """
    height, width = clip.shape[1:3]
    crop_h, crop_w = int(height * scale), int(width * scale)
    top, left = int((height - crop_h) * y_ratio), int((width - crop_w) * x_ratio)
    return np.stack([cv2.resize(frame[top:top + crop_h, left:left + crop_w], (width, height))
                     for frame in clip])


def change_speed(clip: np.ndarray, factor: float) -> np.ndarray:
    """Đổi tốc độ: factor > 1 bỏ bớt frame, factor < 1 lặp frame."""
    n_output = max(1, round(len(clip) / factor))
    return clip[np.linspace(0, len(clip) - 1, n_output).round().astype(int)]


def augment_clip(clip: np.ndarray, config: dict | None = None, seed: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Áp dụng ngẫu nhiên các phép tăng cường.

    config: dict cùng dạng DEFAULT_AUG_CONFIG (None = mặc định). seed: None = không cố định.
    Trả về (clip mới, tên các phép đã dùng). Đổi tốc độ làm đổi số frame nên nhãn theo frame
    phải tính lại; hàm này không đụng tới nhãn hay bbox.
    """
    cfg = DEFAULT_AUG_CONFIG if config is None else config
    rng = np.random.default_rng(seed)
    applied: list[str] = []
    if rng.random() < cfg["hflip"]["p"]:
        clip = horizontal_flip(clip)
        applied.append("hflip")
    params = cfg["brightness_contrast"]
    if rng.random() < params["p"]:
        clip = brightness_contrast(clip, rng.uniform(-params["brightness"], params["brightness"]),
                                   rng.uniform(-params["contrast"], params["contrast"]))
        applied.append("brightness_contrast")
    if rng.random() < cfg["blur"]["p"]:
        clip = blur(clip, int(rng.choice(cfg["blur"]["kernel_sizes"])))
        applied.append("blur")
    if rng.random() < cfg["crop"]["p"]:
        clip = crop_resize(clip, rng.uniform(cfg["crop"]["min_scale"], 1.0), rng.random(), rng.random())
        applied.append("crop")
    params = cfg["speed"]
    if rng.random() < params["p"]:
        clip = change_speed(clip, rng.uniform(params["min_factor"], params["max_factor"]))
        applied.append("speed")
    return clip, applied


def save_augment_preview(clip: np.ndarray, out_path: str | Path, dpi: int = 200) -> Path:
    """Vẽ frame gốc và kết quả từng phép, dùng cho Chương 2. clip là T×H×W×3 BGR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fast = change_speed(clip, PREVIEW_PARAMS["tăng tốc 1.25 lần"]["factor"])
    panels = {"gốc": clip, "lật ngang": horizontal_flip(clip),
              "đổi sáng/tương phản": brightness_contrast(clip, **PREVIEW_PARAMS["đổi sáng/tương phản"]),
              "làm mờ": blur(clip, **PREVIEW_PARAMS["làm mờ"]),
              "cắt xén nhẹ": crop_resize(clip, **PREVIEW_PARAMS["cắt xén nhẹ"]),
              f"tăng tốc 1.25 lần ({len(clip)}→{len(fast)} frame)": fast}
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for axis, (title, shown) in zip(axes.ravel(), panels.items()):
        axis.imshow(cv2.cvtColor(shown[len(shown) // 2], cv2.COLOR_BGR2RGB))
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return Path(out_path)
