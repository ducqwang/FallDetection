"""Sinh bộ dữ liệu GIẢ có cấu trúc giống hệt /kaggle/input để chạy thử prepare_data từ đầu đến cuối.

Đầu vào: thư mục đích, tùy chọn 2 CSV nhãn thật của URFD. Đầu ra: cây thư mục URFD + Le2i và data.yaml đã chỉnh.
Không có CSV thật thì sinh CSV giả. Ảnh là symlink tới một PNG, video là khối màu ngắn, nên chỉ kiểm tra được
luồng xử lý, KHÔNG kiểm tra được chất lượng mô hình trên dữ liệu thật.
Ví dụ: python tests/fake_kaggle.py /tmp/fake --urfd-falls urfall-cam0-falls.csv --urfd-adls urfall-cam0-adls.csv
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
# Số ảnh của 18 ADL CSV_THIEU_FRAME, lấy từ output của check_urfd_structure_v2 trên Kaggle.
ADL_IMAGE_COUNTS = {1: 150, 2: 180, 6: 230, 8: 180, 13: 265, 15: 275, 16: 240, 17: 230, 18: 265, 19: 250,
                    20: 270, 21: 280, 30: 400, 36: 340, 37: 350, 38: 345, 39: 270, 40: 330}
SHORT_FALLS = {13: 12, 15: 14, 16: 9, 17: 7, 21: 5, 25: 11, 27: 12, 28: 10}
ODD_KEYS = {("coffee_room", "01", 26), ("coffee_room", "02", 50), ("coffee_room", "02", 52)}
LE2I_PARTS = [("Coffee_room_01", "Annotation_files", "coffee_room", "01", range(1, 49), 25),
              ("Coffee_room_02", "Annotations_files", "coffee_room", "02", range(49, 71), 25),
              ("Home_01", "Annotation_files", "home", "01", range(1, 31), 24),
              ("Home_02", "Annotation_files", "home", "02", range(31, 61), 24)]
LE2I_ADL = {("coffee_room", "02"): range(63, 71), ("home", "02"): range(38, 61)}
N_SMALL = {("home", "01"): 19, ("home", "02"): 6}
N_LE2I_FRAMES = 30


def fake_urfd_csvs(folder: Path) -> tuple[Path, Path]:
    """Sinh 2 CSV giả có cùng tính chất đã kiểm tra thật (mục 2.1), không có dòng tiêu đề."""
    rows = {"falls": [], "adls": []}
    extra = [0.0] * 8
    for number in range(1, 31):
        n_lying = SHORT_FALLS.get(number, 40)
        labels = [-1] * 20 + [0] * 30 + [1] * n_lying
        rows["falls"] += [[f"fall-{number:02d}", index + 1, label, *extra] for index, label in enumerate(labels)]
    for number in range(1, 41):
        n_images = ADL_IMAGE_COUNTS.get(number, 100)
        frames = list(range(1, n_images + 1))
        if number in ADL_IMAGE_COUNTS:
            frames = frames[(5 if number != 40 else 44):]
            frames = frames[:-47] if number == 17 else frames
        hard = number <= 16
        for index, frame in enumerate(frames):
            label = -1 if not hard or index < 20 else (0 if index < 40 else 1)
            rows["adls"].append([f"adl-{number:02d}", frame, label, *extra])
    paths = folder / "urfall-cam0-falls.csv", folder / "urfall-cam0-adls.csv"
    for path, key in zip(paths, ("falls", "adls")):
        pd.DataFrame(rows[key]).to_csv(path, header=False, index=False)
    return paths


def image_counts(falls_csv: Path, adls_csv: Path) -> dict[str, int]:
    """Số ảnh của từng chuỗi: bằng số dòng CSV, trừ 18 ADL thiếu frame."""
    counts: dict[str, int] = {}
    for path in (falls_csv, adls_csv):
        counts.update(pd.read_csv(path, header=None, usecols=[0])[0].value_counts().to_dict())
    counts.update({f"adl-{number:02d}": n_images for number, n_images in ADL_IMAGE_COUNTS.items()})
    return counts


def make_urfd(root: Path, counts: dict[str, int]) -> Path:
    """Tạo 70 thư mục chuỗi; mọi ảnh là symlink tới 1 PNG 640×480."""
    images = root / "shahliza27" / "UR_fall_detection_dataset_cam0_rgb"
    images.mkdir(parents=True)
    base_png = root / "base.png"
    Image.new("RGB", (640, 480), (90, 120, 150)).save(base_png)
    for seq, n_images in counts.items():
        folder = images / f"{seq}-cam0-rgb"
        folder.mkdir()
        for frame in range(1, n_images + 1):
            (folder / f"{seq}-cam0-rgb-{frame:03d}.png").symlink_to(base_png)
    return images


def _fake_avi(path: Path, fps: int, size: tuple[int, int]) -> None:
    """Video màu đơn N_LE2I_FRAMES frame, codec mjpeg, có track audio PCM (mp3 trong AVI làm ffmpeg tạo packet rỗng)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=gray:s={size[0]}x{size[1]}:r={fps}",
           "-f", "lavfi", "-i", "sine=frequency=440", "-frames:v", str(N_LE2I_FRAMES), "-c:v", "mjpeg",
           "-c:a", "pcm_s16le", "-shortest", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _annotation_lines(kind: str, width: int) -> list[str]:
    """Annotation đúng định dạng thật: đầu video đứng, sau đó nằm; dòng đầu bbox toàn 0."""
    header = {"fall": ["10", "20"], "adl": ["0", "0"], "odd": []}[kind]
    boxes = ["1,1,0,0,0,0"]
    for frame in range(2, N_LE2I_FRAMES + 1):
        lying = kind == "fall" and frame > 15
        corners = (100, 120, 180, 165) if lying else (120, 40, 160, 160)
        boxes.append(f"{frame},1,{corners[0]},{corners[1]},{min(corners[2] + 5, width + 3)},{corners[3]}")
    return header + boxes


def make_le2i(root: Path) -> tuple[Path, int, int]:
    """Tạo cây Le2i đúng số video từng scene. Trả về (thư mục gốc, số dòng bbox, số dòng toàn 0)."""
    dataset = root / "tuyenldvn" / "falldataset-imvia"
    n_rows = n_zero = 0
    for folder, annot_dir, scene, part, numbers, fps in LE2I_PARTS:
        adl_numbers = LE2I_ADL.get((scene, part), range(0))
        n_small = N_SMALL.get((scene, part), 0)
        for order, number in enumerate(numbers):
            size = (320, 180) if order < n_small else (320, 240)
            kind = "odd" if (scene, part, number) in ODD_KEYS else ("adl" if number in adl_numbers else "fall")
            _fake_avi(dataset / folder / folder / "Videos" / f"video ({number}).avi", fps, size)
            lines = _annotation_lines(kind, size[0])
            annotation = dataset / folder / folder / annot_dir / f"video ({number}).txt"
            annotation.parent.mkdir(parents=True, exist_ok=True)
            annotation.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if kind != "odd":
                n_rows, n_zero = n_rows + N_LE2I_FRAMES, n_zero + 1
    for folder, inner, count in (("Lecture_room", "Lecture room", 27), ("Office", "Office", 33)):
        for number in range(1, count + 1):
            _fake_avi(dataset / folder / inner / f"video ({number}).avi", 25, (320, 240))
    (dataset / "README.txt").write_text("fake\n", encoding="utf-8")
    return dataset, n_rows, n_zero


def build(root: Path, falls_csv: Path | None = None, adls_csv: Path | None = None) -> Path:
    """Sinh toàn bộ dữ liệu giả, trả về đường dẫn data.yaml dùng cho prepare_data."""
    labels_dir = root / "input" / "minhquang2701"
    labels_dir.mkdir(parents=True)
    if falls_csv is None:
        new_falls, new_adls = fake_urfd_csvs(labels_dir)
    else:
        new_falls, new_adls = labels_dir / falls_csv.name, labels_dir / adls_csv.name
        new_falls.write_bytes(falls_csv.read_bytes())
        new_adls.write_bytes(adls_csv.read_bytes())
    counts = image_counts(new_falls, new_adls)
    make_urfd(root / "input", counts)
    _, n_rows, n_zero = make_le2i(root / "input")
    config = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    config["paths"].update(input_root=str(root / "input"), out_root=str(root / "out"))
    urfd = config["expect"]["urfd"]
    urfd.update(n_images_min=int(min(counts.values())), n_images_max=int(max(counts.values())))
    config["expect"]["le2i"].update(n_bbox_rows=n_rows, n_bbox_zero_rows=n_zero,
                                    fall_len_frames={"min": 10, "median": 10.0, "max": 10})
    if falls_csv is None:
        urfd["n_adl_hard_neg"] = 16
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh dữ liệu giả có cấu trúc giống Kaggle.")
    parser.add_argument("root", help="thư mục đích, phải chưa tồn tại")
    parser.add_argument("--urfd-falls", help="CSV falls thật (tùy chọn)")
    parser.add_argument("--urfd-adls", help="CSV adls thật (tùy chọn)")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True)
    falls, adls = (Path(args.urfd_falls), Path(args.urfd_adls)) if args.urfd_falls else (None, None)
    print(f"TÓM TẮT | Đã ghi: {build(root, falls, adls)}")


if __name__ == "__main__":
    main()
