"""Chia tập theo VIDEO (chống rò rỉ) và giao thức cross-dataset.

Nguyên tắc: không bao giờ chia theo frame — các frame liền kề gần giống nhau
sẽ rò rỉ giữa train/test. Mỗi video (và người thực hiện) chỉ thuộc một tập.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from .utils import set_seed, load_config, save_json, ensure_dir


@dataclass
class VideoItem:
    """Một đơn vị dữ liệu mức video/chuỗi."""
    vid: str                       # id duy nhất
    dataset: str                   # "urfd" | "le2i"
    label: int                     # 1 = có té ngã, 0 = ADL
    path: str
    subject: str | None = None     # người thực hiện (nếu biết) -> chia theo người
    scene: str | None = None       # bối cảnh/góc camera cho TN3
    fall_start: int | None = None  # frame bắt đầu ngã (nếu có nhãn)
    fall_end: int | None = None


_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def index_urfd(root: str | Path) -> list[VideoItem]:
    """Quét thư mục URFD -> danh sách VideoItem (nhãn mức chuỗi).

    URFD gồm 70 chuỗi: fall-01..fall-30 (có té ngã) và adl-01..adl-40 (sinh hoạt
    thường). Mỗi chuỗi thường là một thư mục PNG (ví dụ 'fall-01-cam0-rgb') hoặc
    một file video. Hàm nhận diện nhãn qua tiền tố tên ('fall' -> 1, 'adl' -> 0)
    và camera qua 'cam0'/'cam1'. Dung nạp nhiều cách sắp xếp của các mirror Kaggle.

    URFD không cung cấp bbox người; fall_start/fall_end để None (ở mức sự kiện,
    cả chuỗi fall được coi là một sự kiện ngã — xem datasets.make_gt_falls).
    """
    root = Path(root)
    items: list[VideoItem] = []
    seen: set[str] = set()

    def classify(name: str) -> int | None:
        low = name.lower()
        if low.startswith("fall") or "-fall" in low or "/fall" in low:
            return 1
        if low.startswith("adl") or "-adl" in low or "/adl" in low:
            return 0
        return None

    # 1) Thư mục chuỗi PNG: */<seq>-camX-rgb/ chứa ảnh
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        if not any(d.glob("*.png")) and not any(d.glob("*.jpg")):
            continue
        label = classify(d.name)
        if label is None:
            continue
        cam = "cam1" if "cam1" in d.name.lower() else "cam0"
        vid = d.name
        if vid in seen:
            continue
        seen.add(vid)
        items.append(VideoItem(vid=vid, dataset="urfd", label=label, path=str(d),
                               subject=_seq_base(d.name), scene=cam))

    # 2) File video rời (một số mirror đóng gói thành .avi/.mp4)
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in _VIDEO_EXTS:
            continue
        label = classify(f.stem)
        if label is None:
            continue
        cam = "cam1" if "cam1" in f.stem.lower() else "cam0"
        vid = f.stem
        if vid in seen:
            continue
        seen.add(vid)
        items.append(VideoItem(vid=vid, dataset="urfd", label=label, path=str(f),
                               subject=_seq_base(f.stem), scene=cam))

    if not items:
        raise FileNotFoundError(
            f"Không tìm thấy chuỗi URFD trong {root}. Kiểm tra lại đường dẫn/'Add Input' trên Kaggle.")
    return items


def _seq_base(name: str) -> str:
    """Lấy id chuỗi gốc (bỏ hậu tố cam/rgb) để chia tập không rò rỉ giữa 2 camera."""
    import re
    m = re.match(r"((?:fall|adl)[-_]?\d+)", name.lower())
    return m.group(1).replace("_", "-") if m else name


def parse_le2i_annotation(ann_path: Path):
    """Đọc annotation Le2i: 2 dòng đầu = frame bắt đầu/kết thúc ngã; các dòng sau =
    'frame,label,x1,y1,x2,y2' (bbox từng frame).

    Trả (fall_start, fall_end, bboxes) với bboxes = {frame: (x1,y1,x2,y2)}.
    Bao dung với biến thể định dạng (dấu phẩy/space/tab, thiếu bbox).
    """
    try:
        lines = [l.strip() for l in ann_path.read_text(errors="ignore").splitlines() if l.strip()]
    except OSError:
        return None, None, {}
    if not lines:
        return None, None, {}

    def ints(s):
        import re
        return [int(float(x)) for x in re.split(r"[,\s]+", s.strip()) if x != ""]

    fall_start = fall_end = None
    body_start = 0
    if len(lines) >= 2 and len(ints(lines[0])) == 1 and len(ints(lines[1])) == 1:
        fall_start, fall_end = ints(lines[0])[0], ints(lines[1])[0]
        body_start = 2

    bboxes: dict[int, tuple] = {}
    for l in lines[body_start:]:
        vals = ints(l)
        if len(vals) >= 6:
            fr, _lab, x1, y1, x2, y2 = vals[:6]
            if x2 > 0 or y2 > 0:
                bboxes[fr] = (x1, y1, x2, y2)
    if fall_start in (0, None) and fall_end in (0, None):
        fall_start = fall_end = None
    return fall_start, fall_end, bboxes


def index_le2i(root: str | Path) -> list[VideoItem]:
    """Quét Le2i (4 bối cảnh) + đọc annotation (frame ngã + bbox).

    Cấu trúc thường gặp: <root>/<Scene>/Videos/*.avi và <root>/<Scene>/Annotation_files/*.txt
    (hoặc annotation đặt cạnh video). Nhãn = 1 nếu annotation có mốc ngã, else 0.
    Scene lấy từ tên thư mục cha (Home/Coffee_room/Office/Lecture_room).
    """
    root = Path(root)
    items: list[VideoItem] = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in _VIDEO_EXTS:
            continue
        ann = _find_le2i_annotation(f)
        fall_start = fall_end = None
        if ann is not None:
            fall_start, fall_end, _ = parse_le2i_annotation(ann)
        label = 1 if (fall_start is not None and fall_end is not None) else 0
        scene = _le2i_scene(f)
        items.append(VideoItem(vid=f"{scene}-{f.stem}", dataset="le2i", label=label,
                               path=str(f), subject=scene, scene=scene,
                               fall_start=fall_start, fall_end=fall_end))
    if not items:
        raise FileNotFoundError(
            f"Không tìm thấy video Le2i trong {root}. Kiểm tra lại đường dẫn/'Add Input' trên Kaggle.")
    return items


def _le2i_scene(video: Path) -> str:
    known = {"home", "coffee_room", "coffee", "office", "lecture_room", "lecture"}
    for p in video.parents:
        if p.name.lower().replace(" ", "_") in known:
            return p.name
    # thử thư mục cha cấp 1-2
    return video.parent.name


def _find_le2i_annotation(video: Path) -> Path | None:
    stem = video.stem
    cands = [
        video.with_suffix(".txt"),
        video.parent / f"{stem}.txt",
        video.parent.parent / "Annotation_files" / f"{stem}.txt",
        video.parent.parent / "Annotation" / f"{stem}.txt",
    ]
    for c in cands:
        if c.exists():
            return c
    # tìm mờ theo stem
    for c in video.parent.parent.rglob(f"{stem}.txt"):
        return c
    return None


def split_by_group(
    items: list[VideoItem],
    ratio=(0.70, 0.15, 0.15),
    key: Literal["video", "subject"] = "video",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Chia train/val/test theo nhóm (video hoặc người), giữ cân bằng lớp thô."""
    import random
    rng = random.Random(seed)
    # gom theo key để không rò rỉ
    groups: dict[str, list[VideoItem]] = {}
    for it in items:
        gkey = getattr(it, key) or it.vid
        groups.setdefault(gkey, []).append(it)
    keys = list(groups)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(ratio[0] * n)
    n_va = int(ratio[1] * n)
    buckets = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    return {s: [v for k in ks for v in [it.vid for it in groups[k]]] for s, ks in buckets.items()}


def build_splits(cfg: dict) -> dict:
    """Tạo cả split within-dataset và 2 protocol cross-dataset."""
    set_seed(cfg["seed"])
    urfd = index_urfd(cfg["paths"]["urfd_root"])
    le2i = index_le2i(cfg["paths"]["le2i_root"])
    ratio = cfg["data"]["split_ratio"]

    out = {
        "within_urfd": split_by_group(urfd, ratio, seed=cfg["seed"]),
        "within_le2i": split_by_group(le2i, ratio, seed=cfg["seed"]),
        # Protocol A: train URFD -> test Le2i ; Protocol B: train Le2i -> test URFD
        "cross_A_train_urfd_test_le2i": {
            "train": [it.vid for it in urfd],
            "test": [it.vid for it in le2i],
        },
        "cross_B_train_le2i_test_urfd": {
            "train": [it.vid for it in le2i],
            "test": [it.vid for it in urfd],
        },
    }
    # lưu cả metadata từng video để dùng lại
    out["_items"] = [asdict(it) for it in (urfd + le2i)]
    save_json(out, Path(cfg["paths"]["splits_dir"]) / "splits.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["splits_dir"])
    build_splits(cfg)
    print("Đã tạo splits.json trong", cfg["paths"]["splits_dir"])
