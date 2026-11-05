"""Đọc Le2i: tìm scene, ghép video với annotation, phân loại fall/ADL, chuẩn hóa bbox.

Đầu vào: thư mục chứa Coffee_room_*, Home_*, Office, Lecture_room.
Đầu ra: danh sách video (khóa scene|phần|N) và bảng bbox frame_idx từ 0, đã clip vào khung hình.
Định dạng annotation đã xác nhận bằng ảnh ghép (mục 2.2 của prompt); README gốc mô tả sai.
Ví dụ: videos = scan_videos(root); info = parse_annotation(videos[0]["annotation"]).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import GT_BBOX_COLUMNS, DataCheckError

SCENE_RE = re.compile(r"^(coffee|home|office|lecture)[ _\-]*(?:room)?[ _\-]*(\d+)?$", re.I)
SCENE_NAMES = {"coffee": "coffee_room", "home": "home", "office": "office", "lecture": "lecture_room"}
USED_SCENES = {"coffee_room", "home"}
VIDEO_SUFFIX = ".avi"
BBOX_COLUMNS_IN_FILE = 6


def find_le2i_root(input_root: Path) -> Path:
    """Tìm thư mục có nhiều thư mục con mang tên scene nhất."""
    best, best_count = None, 0
    for current, subdirs, _ in os.walk(input_root):
        count = sum(1 for name in subdirs if SCENE_RE.match(name.strip()))
        if count > best_count:
            best, best_count = Path(current), count
    if best is None:
        raise FileNotFoundError(f"Không thấy thư mục scene Le2i trong {input_root}")
    return best


def _scene_of(path: Path, root: Path) -> tuple[str, str]:
    """Trả (scene, phần) theo tổ tiên gần nhất có tên scene, ví dụ ('home', '02')."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        match = SCENE_RE.match(part.strip())
        if match:
            return SCENE_NAMES[match.group(1).lower()], match.group(2) or ""
    raise DataCheckError(f"Không xác định được scene của {path}")


def _file_number(stem: str, path: Path) -> int:
    """Lấy N trong 'video (N)'."""
    match = re.search(r"\((\d+)\)", stem)
    if match is None:
        raise DataCheckError(f"Tên file không có dạng 'video (N)': {path}")
    return int(match.group(1))


def _key(path: Path, root: Path) -> str:
    """Khóa ghép (scene|phần|N): N có thể trùng giữa các scene nên phải gồm cả ba."""
    scene, part = _scene_of(path, root)
    return f"{scene}|{part}|{_file_number(path.stem, path)}"


def _is_annotation(path: Path, root: Path) -> bool:
    """Thư mục lúc Annotation_files lúc Annotations_files nên khớp regex 'annot'."""
    return path.suffix.lower() == ".txt" and any(
        re.search(r"annot", part, re.I) for part in path.relative_to(root).parts[:-1])


def scan_videos(root: Path) -> list[dict]:
    """Liệt kê mọi video và annotation tương ứng.

    Trả về list dict: key, scene, part, number, path, annotation (Path hoặc None).
    Annotation không có video hoặc khóa bị trùng thì raise.
    """
    files = [Path(cur) / name for cur, _, names in os.walk(root) for name in names]
    annotations: dict[str, Path] = {}
    for path in (p for p in files if _is_annotation(p, root)):
        key = _key(path, root)
        if key in annotations:
            raise DataCheckError(f"Annotation trùng khóa {key}: {path}")
        annotations[key] = path
    videos: dict[str, dict] = {}
    for path in sorted(p for p in files if p.suffix.lower() == VIDEO_SUFFIX):
        key = _key(path, root)
        if key in videos:
            raise DataCheckError(f"Video trùng khóa {key}: {path}")
        scene, part = _scene_of(path, root)
        videos[key] = {"key": key, "scene": scene, "part": part, "path": path,
                       "number": _file_number(path.stem, path), "annotation": annotations.get(key)}
    orphans = sorted(set(annotations) - set(videos))
    if orphans:
        raise DataCheckError(f"Annotation không có video: {orphans[:5]}")
    return [videos[key] for key in sorted(videos)]


def _is_single_number(line: str) -> bool:
    """Dòng chỉ có một số nguyên không dấu, không có dấu phẩy (dòng bbox có 6 cột)."""
    return "," not in line and re.fullmatch(r"\d+", line) is not None


def _parse_bbox_line(line: str, path: Path) -> list[float]:
    """Đọc 'frame,aux_code,x1,y1,x2,y2'. Trả về 6 số; sai định dạng thì raise kèm tên file."""
    values = line.split(",")
    if len(values) != BBOX_COLUMNS_IN_FILE:
        raise DataCheckError(f"{path}: dòng bbox không đủ 6 cột: {line!r}")
    try:
        numbers = [float(value) for value in values]
    except ValueError as error:
        raise DataCheckError(f"{path}: dòng bbox có giá trị không phải số: {line!r}") from error
    if not (numbers[0].is_integer() and numbers[1].is_integer()):
        raise DataCheckError(f"{path}: frame hoặc aux_code không phải số nguyên: {line!r}")
    return numbers


def _parse_body(lines: list[str], path: Path) -> np.ndarray:
    """Đọc các dòng bbox thành ndarray N×6: frame gốc từ 1, aux_code, x1, y1, x2, y2."""
    return np.array([_parse_bbox_line(line, path) for line in lines], dtype=float).reshape(-1, BBOX_COLUMNS_IN_FILE)


def _parse_top_header(numbered: list[tuple[int, str]], path: Path) -> dict:
    """Đọc file có hai dòng đầu là số: '0','0' là ADL, còn lại là khoảng ngã."""
    first, second = int(numbered[0][1]), int(numbered[1][1])
    kind, start, end = "adl", None, None
    if first != 0 or second != 0:
        if not 0 < first <= second:
            raise DataCheckError(f"{path}: hai dòng đầu không hợp lệ: {first}, {second}")
        kind, start, end = "fall", first, second
    boxes = _parse_body([line for _, line in numbered[2:]], path)
    return {"kind": kind, "start": start, "end": end, "boxes": boxes, "label_source": "header_top",
            "header_lines": (numbered[0][0], numbered[1][0]), "last_frame": int(boxes[:, 0].max()) if len(boxes) else 0}


def _parse_mid_pair(numbered: list[tuple[int, str]], path: Path) -> dict:
    """Đọc file không có hai dòng đầu nhưng có ĐÚNG MỘT cặp dòng đơn-số liền nhau ở giữa hoặc cuối file.

    Cần 0 < start < end <= frame lớn nhất, và sau khi bỏ cặp thì frame liên tục từ 1. Sai thì raise, không đoán.
    """
    singles = [index for index, (_, line) in enumerate(numbered) if _is_single_number(line)]
    if len(singles) != 2 or singles[1] != singles[0] + 1:
        where = [numbered[index][0] for index in singles]
        raise DataCheckError(f"{path}: không có hai dòng đầu và không có đúng một cặp dòng đơn-số liền nhau; "
                             f"dòng đơn-số ở các dòng {where}")
    first, second = singles
    start, end = int(numbered[first][1]), int(numbered[second][1])
    boxes = _parse_body([line for index, (_, line) in enumerate(numbered) if index not in singles], path)
    frames = boxes[:, 0].astype(int)
    if not 0 < start < end <= frames.max():
        raise DataCheckError(f"{path}: cặp ngã không hợp lệ: {start}, {end}, frame lớn nhất {frames.max()}")
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise DataCheckError(f"{path}: frame không liên tục từ 1 sau khi bỏ cặp ngã")
    return {"kind": "fall", "start": start, "end": end, "boxes": boxes, "label_source": "header_mid",
            "header_lines": (numbered[first][0], numbered[second][0]), "last_frame": int(frames.max())}


def parse_annotation(path: Path) -> dict:
    """Đọc 1 file annotation.

    Trả về dict: kind ('fall'|'adl'); start, end (frame gốc từ 1, None với ADL); label_source
    ('header_top': hai dòng đầu, 'header_mid': cặp ở giữa/cuối file); header_lines (số dòng trong file, từ 1);
    last_frame (frame gốc lớn nhất); boxes (ndarray N×6: frame gốc từ 1, aux_code, x1, y1, x2, y2, chưa lọc).
    Mọi định dạng khác đều raise DataCheckError kèm tên file.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    numbered = [(number, line.strip()) for number, line in enumerate(raw_lines, start=1) if line.strip()]
    if len(numbered) < 2:
        raise DataCheckError(f"{path}: file annotation quá ngắn ({len(numbered)} dòng)")
    if _is_single_number(numbered[0][1]) and _is_single_number(numbered[1][1]):
        return _parse_top_header(numbered, path)
    return _parse_mid_pair(numbered, path)


def clean_boxes(video_id: str, boxes: np.ndarray, width: int, height: int,
                n_frames: int) -> tuple[pd.DataFrame, dict]:
    """Chuẩn hóa bbox của 1 video: bỏ dòng toàn 0, về frame_idx từ 0, clip vào [0,w]×[0,h].

    boxes là ndarray N×6 từ parse_annotation. aux_code giữ nguyên cột 2 gốc, chưa rõ nghĩa.

    bbox vượt khung là hợp lệ (người bị cắt ở mép). Sau khi clip, hộp không còn diện tích
    thì bỏ và đếm vào n_degenerate. Trả về (bảng theo GT_BBOX_COLUMNS, thống kê số dòng).
    """
    zero_mask = (boxes[:, 2:] == 0).all(axis=1)
    kept = boxes[~zero_mask]
    frame_idx = kept[:, 0].astype(int) - 1
    if len(frame_idx) and (frame_idx.min() < 0 or frame_idx.max() >= n_frames):
        raise DataCheckError(f"{video_id}: frame bbox ngoài [0, {n_frames}): "
                             f"{frame_idx.min()}..{frame_idx.max()}")
    table = pd.DataFrame({
        "video_id": video_id, "frame_idx": frame_idx,
        "x1": np.clip(kept[:, 2], 0, width), "y1": np.clip(kept[:, 3], 0, height),
        "x2": np.clip(kept[:, 4], 0, width), "y2": np.clip(kept[:, 5], 0, height),
        "aux_code": kept[:, 1].astype(int),
    }, columns=GT_BBOX_COLUMNS)
    valid = (table["x2"] > table["x1"]) & (table["y2"] > table["y1"])
    stats = {"n_rows": len(boxes), "n_zero": int(zero_mask.sum()), "n_degenerate": int((~valid).sum())}
    return table[valid].reset_index(drop=True), stats
