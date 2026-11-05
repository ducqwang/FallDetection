"""Kiểm tra bắt buộc: so số liệu đo được với số liệu đã xác nhận ở configs/data.yaml.

Đầu vào: dữ liệu đã quét và khối `expect`. Đầu ra: list chuỗi mô tả từng chỗ lệch.
prepare_data sẽ dừng nếu danh sách khác rỗng, đúng quy tắc "lệch thì báo TV1, không tự đoán".
Ví dụ: issues = check_urfd(sequences, cfg["expect"]["urfd"]).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _differs(name: str, actual, expected) -> list[str]:
    """Trả về 1 dòng mô tả nếu hai giá trị khác nhau."""
    return [] if actual == expected else [f"{name}: đo được {actual}, mong đợi {expected}"]


def _check_urfd_falls(falls: list[dict], expect: dict) -> list[str]:
    """Mọi Fall khớp hoàn toàn, có đủ nhãn 0 và 1, nhãn 0 dài đúng số frame quy ước."""
    issues = []
    length = expect["fall_label0_len"]
    for seq in falls:
        name = seq["seq"]
        issues += _differs(f"{name} trạng thái", seq["status"], "OK")
        if seq["first_label0"] is None or seq["first_label1"] is None:
            issues.append(f"{name}: thiếu nhãn 0 hoặc 1")
            continue
        issues += _differs(f"{name} số frame nhãn 0", seq["n_label0"], length)
        issues += _differs(f"{name} fall_end - fall_start", seq["first_label1"] - seq["first_label0"], length)
    short = sorted(s["seq"] for s in falls if s["n_label1"] < expect["short_after_lying_frames"])
    return issues + _differs("Fall còn rất ít frame sau khi nằm", short, sorted(expect["fall_short_after_lying"]))


def check_urfd(sequences: list[dict], expect: dict) -> list[str]:
    """So toàn bộ URFD với mục 2.1. Trả về list chỗ lệch (rỗng nếu khớp)."""
    falls = [s for s in sequences if s["label"] == "fall"]
    adls = [s for s in sequences if s["label"] == "adl"]
    issues = _differs("số chuỗi Fall", len(falls), expect["n_fall"])
    issues += _differs("số chuỗi ADL", len(adls), expect["n_adl"])
    issues += _differs("camera", sorted({s["camera"] for s in sequences}), ["cam0"])
    issues += _differs("kích thước ảnh", sorted({(s["width"], s["height"]) for s in sequences}),
                       [tuple(expect["size"])])
    counts = [s["n_images"] for s in sequences]
    issues += _differs("số ảnh ít nhất", min(counts), expect["n_images_min"])
    issues += _differs("số ảnh nhiều nhất", max(counts), expect["n_images_max"])
    for seq in sequences:
        contiguous = seq["numbers"] == list(range(1, len(seq["numbers"]) + 1))
        if not contiguous:
            issues.append(f"{seq['seq']}: số frame ảnh không liên tục từ 1")
        issues += _differs(f"{seq['seq']} dòng CSV không có ảnh", seq["n_csv_without_image"], 0)
    issues += _check_urfd_falls(falls, expect)
    missing = sorted(s["seq"] for s in sequences if s["status"] == "CSV_THIEU_FRAME")
    issues += _differs("chuỗi CSV_THIEU_FRAME", missing, sorted(expect["adl_csv_thieu_frame"]))
    n_hard = sum(1 for s in adls if s["n_label0"] + s["n_label1"] > 0)
    return issues + _differs("số ADL negative khó", n_hard, expect["n_adl_hard_neg"])


def _check_le2i_parts(used: list[dict], expect: dict) -> list[str]:
    """Bảng số video theo scene/phần, fps và số video 320×180."""
    issues = []
    for part_key, numbers in expect["per_part"].items():
        rows = [v for v in used if f"{v['scene']}|{v['part']}" == part_key]
        kinds = Counter(v["kind"] for v in rows)
        actual = [len(rows), kinds["fall"], kinds["adl"]]
        issues += _differs(f"{part_key} [video, fall, adl]", actual, list(numbers))
        small_size = tuple(expect["small_size"])
        n_small = sum(1 for v in rows if (v["width"], v["height"]) == small_size)
        issues += _differs(f"{part_key} số video {small_size}", n_small, expect["n_small_size"].get(part_key, 0))
    other_sizes = {(v["width"], v["height"]) for v in used} - {tuple(size) for size in expect["sizes"]}
    issues += _differs("kích thước lạ", sorted(other_sizes), [])
    for video in used:
        issues += _differs(f"{video['key']} fps", video["fps"], float(expect["fps"][video["scene"]]))
    return issues


def _check_le2i_bbox(used: list[dict], expect: dict) -> list[str]:
    """Số dòng bbox và độ dài khoảng ngã, chỉ trên video header_top vì số mong đợi đo trên nhóm này."""
    top = [v for v in used if v["label_source"] == "header_top"]
    n_rows = sum(v["bbox_stats"]["n_rows"] for v in top)
    n_zero = sum(v["bbox_stats"]["n_zero"] for v in top)
    issues = _differs("số dòng bbox", n_rows, expect["n_bbox_rows"])
    issues += _differs("số dòng bbox toàn 0", n_zero, expect["n_bbox_zero_rows"])
    lengths = [v["end"] - v["start"] for v in top if v["kind"] == "fall"]
    if not lengths:
        return issues + ["không có video fall nào để đo khoảng ngã"]
    measured = {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)}
    return issues + _differs("độ dài khoảng ngã (frame)", measured, dict(expect["fall_len_frames"]))


def required_check_lines(sequences: list[dict], videos: list[dict]) -> list[str]:
    """Kết quả các kiểm tra bắt buộc của Việc 1.5, để in ra và ghi vào DATA_CARD.md.

    Mọi con số đo từ dữ liệu vừa quét. fps tính bằng frame/giây, kích thước bằng pixel.
    """
    falls = [s for s in sequences if s["label"] == "fall"]
    lacking = sorted(s["seq"] for s in sequences if s["status"] == "CSV_THIEU_FRAME")
    kinds = Counter(v["kind"] for v in videos)
    fps_counts = Counter(f"{v['scene']} {v['fps']:g} fps" for v in videos)
    size_counts = Counter(f"{v['width']}×{v['height']}" for v in videos)
    return [
        f"URFD Fall khớp hoàn toàn: {sum(s['status'] == 'OK' for s in falls)}/{len(falls)}",
        f"URFD ADL CSV_THIEU_FRAME: {len(lacking)} chuỗi: {', '.join(lacking)}",
        f"URFD dòng CSV không có ảnh: {sum(s['n_csv_without_image'] for s in sequences)}",
        f"URFD chuỗi giữ lại: {len(sequences)}, không loại chuỗi nào",
        f"Le2i: {len(videos)} video ({kinds['fall']} fall, {kinds['adl']} adl)",
        f"Le2i has_event_label = 1: {sum(v['row']['has_event_label'] for v in videos)}/{len(videos)}",
        f"Le2i nguồn nhãn: {dict(sorted(Counter(v['label_source'] for v in videos).items()))}",
        f"Le2i fps: {dict(sorted(fps_counts.items()))}",
        f"Le2i kích thước: {dict(sorted(size_counts.items()))}",
    ]


def check_le2i(all_videos: list[dict], used: list[dict], expect: dict) -> list[str]:
    """So Le2i với mục 2.2. Trả về list chỗ lệch (rỗng nếu khớp)."""
    issues = _differs("tổng video trong thư mục", len(all_videos), expect["n_total_folder"])
    for scene, count in expect["excluded"].items():
        rows = [v for v in all_videos if v["scene"] == scene]
        issues += _differs(f"{scene} số video", len(rows), count)
        issues += _differs(f"{scene} số annotation", sum(v["annotation"] is not None for v in rows), 0)
    issues += _differs("video thiếu annotation trong phạm vi dùng",
                       [v["key"] for v in used if v["annotation"] is None], [])
    measured = {v["key"]: {"start": v["start"], "end": v["end"], "lines": list(v["header_lines"]),
                           "last_frame": v["last_frame"]} for v in used if v["label_source"] == "header_mid"}
    issues += _differs("video header_mid", measured, dict(expect["header_mid"]))
    return issues + _check_le2i_parts(used, expect) + _check_le2i_bbox(used, expect)
