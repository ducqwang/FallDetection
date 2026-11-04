"""Đọc UR Fall: tìm dữ liệu, quét chuỗi ảnh, đọc CSV nhãn, ghép theo SỐ FRAME.

Đầu vào: thư mục cha chứa 70 chuỗi cam0 và 2 file CSV không có dòng tiêu đề.
Đầu ra: danh sách dict từng chuỗi, bảng nhãn frame (frame_idx từ 0) và báo cáo đối chiếu.
Ví dụ: labels = read_labels(falls_csv, adls_csv); rows = match_labels(scan_sequences(root), labels).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from .common import DataCheckError

SEQ_DIR_RE = re.compile(r"^(adl|fall)-(\d+)-(cam\d+)-rgb$", re.I)
SEQ_NAME_RE = re.compile(r"(adl|fall)[-_]?(\d+)", re.I)
IMAGE_SUFFIXES = {".png"}
CSV_COLUMNS = ["seq", "frame", "label", "HeightWidthRatio", "MajorMinorRatio",
               "BoundingBoxOccupancy", "MaxStdXZ", "HHmaxRatio", "H", "D", "P40"]
USED_CSV_COLUMNS = ["seq", "frame", "label", "HeightWidthRatio"]


def find_urfd_root(input_root: Path) -> Path:
    """Tìm thư mục có nhiều thư mục con dạng adl-NN-cam0-rgb / fall-NN-cam0-rgb nhất."""
    best, best_count = None, 0
    for current, subdirs, _ in os.walk(input_root):
        count = sum(1 for name in subdirs if SEQ_DIR_RE.match(name))
        if count > best_count:
            best, best_count = Path(current), count
        if count:
            subdirs[:] = []  # đã tới tầng chuỗi, không cần đi sâu thêm
    if best is None:
        raise FileNotFoundError(f"Không thấy thư mục chuỗi URFD trong {input_root}")
    return best


def find_label_csv(input_root: Path, suffix: str) -> Path:
    """Tìm CSV theo ĐUÔI tên ('-falls.csv' / '-adls.csv').

    Không khớp theo chuỗi con 'fall' vì tên 'urfall' cũng chứa chữ đó.
    """
    matches = [Path(cur) / name for cur, _, names in os.walk(input_root) for name in names
               if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cần đúng 1 file *{suffix} trong {input_root}, thấy {len(matches)}: "
                                f"{matches[:5]}. Hãy khai báo trong configs/data.yaml")
    return matches[0]


def normalize_seq(name: str) -> str:
    """Chuẩn hóa tên chuỗi về dạng 'fall-01' hoặc 'adl-07'."""
    match = SEQ_NAME_RE.search(str(name))
    if match is None:
        raise DataCheckError(f"Tên chuỗi không hợp lệ trong CSV: {name!r}")
    return f"{match.group(1).lower()}-{int(match.group(2)):02d}"


def frame_number(path: Path) -> int:
    """Lấy số frame (đánh số từ 1) ở cuối tên ảnh, ví dụ adl-01-cam0-rgb-007.png -> 7."""
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise DataCheckError(f"Tên ảnh không có số frame: {path}")
    return int(numbers[-1])


def scan_sequences(root: Path) -> list[dict]:
    """Quét các thư mục chuỗi; ảnh đọc bằng rglob, không giả định độ sâu.

    Trả về list dict: seq, camera, label (fall|adl), frames (list Path theo số frame tăng dần),
    numbers (list int, số frame gốc từ 1), width, height (pixel).
    """
    sequences = []
    for folder in sorted(root.iterdir()):
        match = SEQ_DIR_RE.match(folder.name)
        if not folder.is_dir() or match is None:
            continue
        images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        pairs = sorted(((frame_number(p), p) for p in images), key=lambda item: item[0])
        if not pairs:
            raise DataCheckError(f"{folder.name}: không có ảnh")
        with Image.open(pairs[0][1]) as image:
            width, height = image.size
        sequences.append({
            "seq": f"{match.group(1).lower()}-{int(match.group(2)):02d}",
            "camera": match.group(3).lower(), "label": match.group(1).lower(),
            "frames": [p for _, p in pairs], "numbers": [n for n, _ in pairs],
            "width": width, "height": height,
        })
    return sequences


def _read_one_csv(path: Path, expected_prefix: str) -> pd.DataFrame:
    """Đọc 1 CSV không có tiêu đề, chỉ giữ 4 cột dùng, assert đúng loại chuỗi."""
    table = pd.read_csv(path, header=None, names=CSV_COLUMNS, usecols=USED_CSV_COLUMNS)
    table["seq"] = table["seq"].map(normalize_seq)
    wrong = sorted(set(table.loc[~table["seq"].str.startswith(expected_prefix), "seq"]))
    if wrong:
        raise DataCheckError(f"{path.name} chứa chuỗi không phải {expected_prefix}-*: {wrong[:5]}")
    bad_labels = sorted(set(table["label"]) - {-1, 0, 1})
    if bad_labels:
        raise DataCheckError(f"{path.name} có nhãn lạ: {bad_labels}")
    return table


def read_labels(falls_csv: Path, adls_csv: Path) -> pd.DataFrame:
    """Đọc 2 CSV nhãn. Cột frame giữ nguyên gốc (từ 1), nhãn -1/0/1."""
    if falls_csv.resolve() == adls_csv.resolve():
        raise DataCheckError(f"Hai đường dẫn CSV trùng nhau: {falls_csv}")
    table = pd.concat([_read_one_csv(falls_csv, "fall"), _read_one_csv(adls_csv, "adl")],
                      ignore_index=True)
    duplicated = table[table.duplicated(["seq", "frame"])]
    if not duplicated.empty:
        raise DataCheckError(f"CSV có frame trùng: {duplicated[['seq', 'frame']].head().to_dict('records')}")
    return table.sort_values(["seq", "frame"]).reset_index(drop=True)


def _match_one(sequence: dict, labels: pd.DataFrame) -> dict:
    """Ghép nhãn của 1 chuỗi theo số frame; không ghép theo thứ tự dòng."""
    numbers = sequence["numbers"]
    label_of = dict(zip(labels["frame"], labels["label"]))
    csv_frames = sorted(label_of)
    missing_images = sorted(set(csv_frames) - set(numbers))
    if not csv_frames:
        status = "THIEU_CSV"
    elif missing_images:
        status = "LECH_THAT"
    else:
        status = "OK" if len(csv_frames) == len(numbers) else "CSV_THIEU_FRAME"
    row = {"status": status, "n_images": len(numbers), "n_csv_rows": len(csv_frames),
           "n_csv_without_image": len(missing_images),
           "first_label0": None, "first_label1": None, "n_label0": 0, "n_label1": 0}
    if csv_frames:
        row.update(miss_head=csv_frames[0] - numbers[0], miss_tail=numbers[-1] - csv_frames[-1],
                   gap_mid=csv_frames[-1] - csv_frames[0] + 1 - len(csv_frames),
                   label_first_frame=csv_frames[0] - 1, label_last_frame=csv_frames[-1] - 1)
        for value, first_key, count_key in ((0, "first_label0", "n_label0"), (1, "first_label1", "n_label1")):
            hits = [frame for frame in csv_frames if label_of[frame] == value]
            row[count_key] = len(hits)
            row[first_key] = hits[0] if hits else None
    return row


def match_labels(sequences: list[dict], labels: pd.DataFrame) -> list[dict]:
    """Thêm vào mỗi chuỗi kết quả đối chiếu ảnh ↔ CSV.

    first_label0/first_label1 vẫn là số frame gốc từ 1; label_first_frame,
    label_last_frame đã về 0-based. CSV nhắc tới chuỗi không có ảnh thì raise.
    """
    known = {item["seq"] for item in sequences}
    orphan = sorted(set(labels["seq"]) - known)
    if orphan:
        raise DataCheckError(f"CSV có chuỗi không có thư mục ảnh: {orphan[:5]}")
    groups = dict(tuple(labels.groupby("seq")))
    empty = labels.iloc[0:0]
    return [{**item, **_match_one(item, groups.get(item["seq"], empty))} for item in sequences]


def frame_label_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Bảng urfd_frame_labels: chỉ frame CÓ trong CSV gốc, frame_idx từ 0 (seq vẫn là tên gốc)."""
    table = labels[["seq", "frame", "label"]].copy()
    table["frame_idx"] = table["frame"] - 1
    return table[["seq", "frame_idx", "label"]]
