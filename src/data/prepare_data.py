"""Việc 1: chuẩn hóa URFD và Le2i thành bộ dữ liệu nhom05-falldet-processed.

Đầu vào: /kaggle/input (tự tìm, ghi đè bằng configs/data.yaml). Đầu ra: videos/*.mp4, metadata.csv,
annotations/{le2i_gt_bbox.csv, urfd_frame_labels.csv}, DATA_CARD.md trong out_root.
Số liệu lệch với mục 2.1, 2.2 thì dừng TRƯỚC khi chuyển video và in từng chỗ lệch.
Ví dụ: python -m src.data.prepare_data --config configs/data.yaml --workers 4
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from . import checks, le2i, media, urfd
from .common import FRAME_LABEL_COLUMNS, GT_BBOX_COLUMNS, DataCheckError, urfd_video_id
from .data_card import build_data_card
from .metadata import le2i_row, metadata_frame, urfd_row


LE2I_FRAME_HINT = " | gợi ý: thử -fps_mode passthrough (hoặc -vsync 0 với ffmpeg cũ)"


def load_urfd(paths: dict, expect: dict) -> tuple[list[dict], pd.DataFrame, list[str]]:
    """Quét URFD. Trả về (chuỗi đã đối chiếu, bảng nhãn gốc, danh sách chỗ lệch)."""
    input_root = Path(paths["input_root"])
    root = Path(paths["urfd_root"]) if paths.get("urfd_root") else urfd.find_urfd_root(input_root)
    falls = Path(paths["urfd_csv_falls"]) if paths.get("urfd_csv_falls") else urfd.find_label_csv(input_root, "-falls.csv")
    adls = Path(paths["urfd_csv_adls"]) if paths.get("urfd_csv_adls") else urfd.find_label_csv(input_root, "-adls.csv")
    print(f"urfd: ảnh {root} | nhãn {falls.name}, {adls.name}")
    labels = urfd.read_labels(falls, adls)
    sequences = urfd.match_labels(urfd.scan_sequences(root), labels)
    return sequences, labels, checks.check_urfd(sequences, expect)


def _bbox_of(video: dict) -> tuple[pd.DataFrame, dict]:
    """Bảng bbox đã chuẩn hóa và thống kê số dòng của 1 video."""
    if video["boxes"] is None:
        return pd.DataFrame(columns=GT_BBOX_COLUMNS), {"n_rows": 0, "n_zero": 0, "n_degenerate": 0}
    return le2i.clean_boxes(video["row"]["video_id"], video["boxes"], video["width"], video["height"], video["n_frames"])


def _read_le2i_video(video: dict) -> dict:
    """Đọc thông số (ffprobe), annotation và bbox của 1 video Le2i.

    Video lỗi (không đọc được hoặc annotation sai định dạng) trả về khóa read_error để main
    in đủ mọi video lỗi một lần rồi mới dừng.
    """
    try:
        info = media.probe_video(video["path"])
        n_frames = media.count_frames(video["path"])
        parsed = {"kind": "no_annot", "start": None, "end": None, "boxes": None, "label_source": ""}
        if video["annotation"] is not None:
            parsed = le2i.parse_annotation(video["annotation"])
        item = {**video, **info, **parsed, "n_frames": n_frames}
        item["row"] = le2i_row(item)
        item["bbox_table"], item["bbox_stats"] = _bbox_of(item)
    except (RuntimeError, DataCheckError) as error:
        return {**video, "read_error": str(error)}
    return item


def load_le2i(paths: dict, expect: dict, workers: int) -> tuple[list[dict], pd.DataFrame, list[str], list[str]]:
    """Quét Le2i, loại Office và Lecture room. Trả về (video dùng, bảng bbox, chỗ lệch, video lỗi)."""
    root = Path(paths["le2i_root"]) if paths.get("le2i_root") else le2i.find_le2i_root(Path(paths["input_root"]))
    print(f"le2i: thư mục gốc {root}")
    all_videos = le2i.scan_videos(root)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_read_le2i_video, [v for v in all_videos if v["scene"] in le2i.USED_SCENES]))
    used = [item for item in results if "read_error" not in item]
    unreadable = [f"{item['key']}: {item['read_error']}" for item in results if "read_error" in item]
    tables = [item["bbox_table"] for item in used if len(item["bbox_table"])]
    boxes = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=GT_BBOX_COLUMNS)
    return used, boxes, checks.check_le2i(all_videos, used, expect), unreadable


def print_inventory(sequences: list[dict], videos: list[dict], unreadable: list[str]) -> None:
    """In báo cáo kiểm tra: số video theo lớp, theo scene, video thiếu annotation, video lỗi."""
    urfd_classes = Counter(s["label"] for s in sequences)
    urfd_status = Counter(s["status"] for s in sequences)
    print(f"urfd: {len(sequences)} video ({urfd_classes['fall']} fall, {urfd_classes['adl']} adl) | {dict(urfd_status)}")
    kinds = Counter(v["kind"] for v in videos)
    sources = dict(Counter(v["label_source"] for v in videos))
    print(f"le2i: {len(videos)} video ({kinds['fall']} fall, {kinds['adl']} adl, "
          f"{kinds['no_annot']} thiếu annotation) | {sources}")
    for key, count in sorted(Counter(f"{v['scene']}|{v['part']}|{v['kind']}" for v in videos).items()):
        print(f"  le2i {key}: {count}")
    print(f"video thiếu annotation: {[v['key'] for v in videos if v['kind'] == 'no_annot']}")
    print(f"video lỗi không đọc được: {unreadable}")


def _verify_mp4(path: Path, expected: dict, name: str, is_le2i: bool) -> str | None:
    """So mp4 với nguồn về số frame, fps và kích thước. Khớp trả về None, lệch trả về mô tả."""
    actual = {**media.probe_video(path), "n_frames": media.count_frames(path)}
    wrong = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
    if not wrong:
        return None
    message = f"{name}: mp4 lệch nguồn (đo được, mong đợi): {wrong}"
    if "n_frames" in wrong:
        message = f"{name}: số frame nguồn {expected['n_frames']}, số frame mp4 {actual['n_frames']}; {message}"
        message += LE2I_FRAME_HINT if is_le2i else ""
    return message


def _convert_one(task: dict) -> str | None:
    """Chuyển 1 video rồi kiểm tra; `--skip-existing` giữ mp4 đã có sau khi kiểm tra lại.

    Trả về None nếu khớp, ngược lại trả về mô tả chỗ lệch.
    """
    out_path: Path = task["out_path"]
    if not (task["skip_existing"] and out_path.exists()):
        if "frames" in task:
            media.png_sequence_to_mp4(task["frames"], out_path, task["fps"])
        else:
            media.avi_to_mp4(task["source"], out_path)
    return _verify_mp4(out_path, {"fps": float(task["fps"]), "n_frames": task["n_frames"], "width": task["width"],
                                  "height": task["height"]}, task["video_id"], "source" in task)


def convert_videos(meta: pd.DataFrame, sequences: list[dict], videos: list[dict], out_root: Path,
                   workers: int, skip_existing: bool) -> tuple[int, int]:
    """Chuyển toàn bộ sang mp4 và kiểm tra từng file.

    Trả về (số video đã chuyển, số video mp4 lệch). Có video lệch thì in từng video rồi raise,
    nên giá trị thứ hai luôn bằng 0 khi hàm trả về bình thường.
    """
    sources = {urfd_video_id(s["seq"], s["camera"]): {"frames": s["frames"]} for s in sequences}
    sources.update({v["row"]["video_id"]: {"source": v["path"]} for v in videos})
    tasks = [{**sources[row.video_id], "video_id": row.video_id, "out_path": out_root / row.path,
              "fps": row.fps, "n_frames": row.n_frames, "width": row.width, "height": row.height,
              "skip_existing": skip_existing} for row in meta.itertuples()]
    with ThreadPoolExecutor(workers) as pool:
        problems = [problem for problem in pool.map(_convert_one, tasks) if problem]
    for problem in problems:
        print(problem)
    if problems:
        raise DataCheckError(f"{len(problems)} video mp4 lệch nguồn, báo TV1:\n" + "\n".join(problems))
    return len(tasks), len(problems)


def run_log(meta: pd.DataFrame, n_mismatch: int) -> str:
    """Dòng log số liệu cuối chạy, đếm từ metadata. n_mismatch là số đếm thật từ convert_videos."""
    urfd_rows, le2i_rows = meta[meta["dataset"] == "urfd"], meta[meta["dataset"] == "le2i"]
    return (f"urfd: {len(urfd_rows)} video ({(urfd_rows['label'] == 'fall').sum()} fall, {(urfd_rows['label'] == 'adl').sum()} adl) | "
            f"le2i: {len(le2i_rows)} video ({(le2i_rows['label'] == 'fall').sum()} fall, {(le2i_rows['label'] == 'adl').sum()} adl) | "
            f"mp4 frame mismatch: {n_mismatch}")


def write_outputs(meta: pd.DataFrame, labels: pd.DataFrame, boxes: pd.DataFrame, card: str, out_root: Path) -> list[Path]:
    """Ghi metadata, annotations và DATA_CARD; trả về danh sách file đã ghi."""
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    frame_labels = urfd.frame_label_table(labels)
    frame_labels["video_id"] = frame_labels["seq"].map(lambda seq: urfd_video_id(seq, "cam0"))
    written = {"metadata.csv": meta, "annotations/le2i_gt_bbox.csv": boxes[GT_BBOX_COLUMNS],
               "annotations/urfd_frame_labels.csv": frame_labels[FRAME_LABEL_COLUMNS]}
    for name, table in written.items():
        table.to_csv(out_root / name, index=False)
    (out_root / "DATA_CARD.md").write_text(card, encoding="utf-8")
    return [out_root / name for name in written] + [out_root / "DATA_CARD.md"]


def parse_args() -> argparse.Namespace:
    """Khai báo tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chuẩn hóa URFD và Le2i (Home + Coffee room).")
    parser.add_argument("--config", default="configs/data.yaml", help="file cấu hình dữ liệu")
    parser.add_argument("--input-root", help="ghi đè paths.input_root")
    parser.add_argument("--out-root", help="ghi đè paths.out_root")
    parser.add_argument("--workers", type=int, default=4, help="số tiến trình ffmpeg chạy song song")
    parser.add_argument("--skip-existing", action="store_true", help="giữ mp4 đã có (vẫn kiểm tra số frame)")
    return parser.parse_args()


def main() -> None:
    """Quét, kiểm tra, chuyển video rồi ghi toàn bộ đầu ra."""
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = {**cfg["paths"], **({"input_root": args.input_root} if args.input_root else {})}
    out_root = Path(args.out_root or paths["out_root"])
    media.require_ffmpeg()
    sequences, labels, urfd_issues = load_urfd(paths, cfg["expect"]["urfd"])
    videos, boxes, le2i_issues, unreadable = load_le2i(paths, cfg["expect"]["le2i"], args.workers)
    print_inventory(sequences, videos, unreadable)
    issues = urfd_issues + le2i_issues + [f"video lỗi {item}" for item in unreadable]
    if issues:
        raise DataCheckError(f"{len(issues)} chỗ lệch so với mô tả, báo TV1:\n" + "\n".join(issues))
    check_lines = checks.required_check_lines(sequences, videos)
    print("kiểm tra số liệu: khớp mục 2.1 và 2.2\n" + "\n".join(check_lines))
    meta = metadata_frame([urfd_row(s, cfg["expect"]["urfd"]["fps"]) for s in sequences] + [v["row"] for v in videos])
    n_ok, n_mismatch = convert_videos(meta, sequences, videos, out_root, args.workers, args.skip_existing)
    bbox_stats = {key: sum(v["bbox_stats"][key] for v in videos) for key in ("n_rows", "n_zero", "n_degenerate")}
    card = build_data_card(meta, sequences, videos, bbox_stats, cfg["dataset_sources"], check_lines,
                           cfg["expect"]["urfd"]["short_after_lying_frames"])
    written = write_outputs(meta, labels, boxes, card, out_root)
    n_skipped = sum(cfg["expect"]["le2i"]["excluded"].values())
    print(run_log(meta, n_mismatch))
    print(f"TÓM TẮT | video đã xử lý: {n_ok} | bỏ qua: {n_skipped} (Office, Lecture room: không có annotation) "
          f"| bbox: {len(boxes)} dòng")
    print("Đã ghi:", ", ".join(str(path) for path in written), f"và {n_ok} mp4 trong {out_root / 'videos'}")


if __name__ == "__main__":
    main()
