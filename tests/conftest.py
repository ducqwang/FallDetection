"""Cấu hình pytest: thêm gốc repo vào sys.path và cung cấp bảng metadata giả dùng chung.

Đầu vào: không có. Đầu ra: fixture cho các test của TV1.
Ví dụ: python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def meta_row(video_id: str, label: str, n_frames: int = 300, fps: float = 30.0, start=None, end=None,
             dataset: str = "urfd", scene: str = "urfd", hard: int = 0, has_label: int = 1) -> dict:
    """1 dòng metadata giả đủ các cột mà bộ đánh giá và phần chia tập cần."""
    return {"video_id": video_id, "dataset": dataset, "scene": scene, "label": label, "fall_start_frame": start,
            "fall_end_frame": end, "fps": fps, "n_frames": n_frames, "duration_s": n_frames / fps,
            "has_event_label": has_label, "adl_hard_neg": hard, "width": 640, "height": 480}


def meta_frame(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    for column in ("fall_start_frame", "fall_end_frame"):
        table[column] = pd.to_numeric(table[column]).astype("Int64")
    return table


@pytest.fixture
def full_meta() -> pd.DataFrame:
    """Metadata có đúng phân bố thật: URFD 30 fall, 24 ADL thường, 16 hard-neg; Le2i như bảng 2.2."""
    rows = [meta_row(f"urfd_fall-{i:02d}_cam0", "fall", start=50, end=80) for i in range(1, 31)]
    rows += [meta_row(f"urfd_adl-{i:02d}_cam0", "adl", hard=int(i <= 16)) for i in range(1, 41)]
    cells = [("coffee_room", "01", "fall", 47), ("coffee_room", "02", "fall", 12), ("coffee_room", "02", "adl", 8),
             ("home", "01", "fall", 30), ("home", "02", "fall", 7), ("home", "02", "adl", 23)]
    number = 0
    for scene, part, label, count in cells:
        for _ in range(count):
            number += 1
            is_fall = label == "fall"
            rows.append(meta_row(f"le2i_{scene}{part}_{number:03d}", label, dataset="le2i", scene=scene, fps=25.0,
                                 start=50 if is_fall else None, end=75 if is_fall else None))
    for index in range(3):
        rows.append(meta_row(f"le2i_coffee_room02_09{index}", None, dataset="le2i", scene="coffee_room", has_label=0))
    return meta_frame(rows)
