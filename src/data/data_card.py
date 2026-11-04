"""Sinh DATA_CARD.md từ số liệu đo được của prepare_data.

Đầu vào: metadata, chuỗi URFD đã đối chiếu, thống kê bbox, cấu hình nguồn dữ liệu.
Đầu ra: văn bản Markdown; mọi con số lấy từ dữ liệu vừa xử lý, không gõ tay.
Ví dụ: Path("DATA_CARD.md").write_text(build_data_card(meta, sequences, stats, sources)).
"""
from __future__ import annotations

import pandas as pd

MISSING_VERSION = "chưa điền trong configs/data.yaml"
LICENSES = """## Giấy phép và trích dẫn

- **URFD:** CC BY-NC. Trích dẫn: B. Kwolek, M. Kępski (2014), *Human fall detection on embedded
  platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine.
- **Le2i:** chỉ dùng cho học thuật. Trích dẫn: I. Charfi, J. Miteran, J. Dubois, M. Atri, R. Tourki,
  *Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost
  based classification*, Journal of Electronic Imaging, 22(4), 2013.
"""
CONVENTIONS = """## Quy ước

- `video_id` không dấu cách, không ngoặc, ví dụ `urfd_fall-01_cam0`, `le2i_home02_031`.
- `frame_idx` bắt đầu từ 0. Frame gốc của URFD và Le2i đánh số từ 1 nên đã trừ 1.
- bbox `x1, y1, x2, y2` theo pixel tuyệt đối trên video gốc.
- Thời gian `t = frame_idx / fps` với fps của từng video. Không giả định fps chung.
- Chia tập theo video, seed 42.
- **URFD: nhãn 0 dài đúng 30 frame (1.0 s) ở mọi chuỗi Fall, nên `fall_end = fall_start + 30`.
  Đây là quy ước của bộ dữ liệu, không phải thời điểm đo được.**
- ADL của URFD chỉ dùng nhãn mức video. Nhãn 1 của ADL không tạo sự kiện ngã.
- Frame không có trong CSV của URFD là chưa có nhãn (NaN), không tự điền -1.
"""
EXCLUSION = """## Phạm vi Le2i

Chỉ dùng Home và Coffee room. **Office và Lecture room bị loại hẳn** (không chuyển mp4, không có trong
`metadata.csv`, `splits.json` và EDA) vì hai scene này không có file annotation: không có khoảng té ngã,
không có bbox, và cũng không dùng làm mẫu ADL được vì không biết video nào có té ngã.
Nguồn tham khảo: arXiv 1812.01923 ("Due to loss of annotation files, we only use the recordings
from home and coffee room").

README của Le2i ghi "320x240 25FPS" và mô tả bbox là chiều cao, chiều rộng, tọa độ tâm. Dữ liệu thực tế
khác: fps và kích thước ở bảng dưới; bbox là góc `x1,y1,x2,y2`.
"""


def _sources_section(sources: dict) -> str:
    lines = ["## Nguồn dữ liệu", "", "| Dataset | URL | Version |", "|---|---|---|"]
    for name, source in sources.items():
        lines.append(f"| {name} | {source['url']} | {source.get('version') or MISSING_VERSION} |")
    return "\n".join(lines) + "\n"


def _counts_section(meta: pd.DataFrame) -> str:
    table = meta.assign(lop=meta["label"].where(meta["label"] != "", "chưa xác định").fillna("chưa xác định"),
                        phan=meta["scene_part"].fillna("")).groupby(
        ["dataset", "scene", "phan", "fps", "width", "height", "lop"]).size().reset_index(name="so_video")
    lines = ["## Số lượng thực tế", "",
             "| Dataset | Scene | Phần | fps | Kích thước | Lớp | Số video |", "|---|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.dataset} | {row.scene} | {row.phan} | {row.fps:g} | {row.width}×{row.height} "
                     f"| {row.lop} | {row.so_video} |")
    n_hard = int(meta["adl_hard_neg"].sum())
    lines += ["", f"Tổng {len(meta)} video. `adl_hard_neg = 1`: {n_hard} chuỗi ADL của URFD "
                  "(cúi, bò, ngồi/nằm giường, nằm sofa; không phải té ngã)."]
    return "\n".join(lines) + "\n"


def _urfd_section(sequences: list[dict]) -> str:
    falls_ok = sum(1 for s in sequences if s["label"] == "fall" and s["status"] == "OK")
    adl_ok = sum(1 for s in sequences if s["label"] == "adl" and s["status"] == "OK")
    lacking = [s for s in sequences if s["status"] == "CSV_THIEU_FRAME"]
    lines = ["## Chỗ lệch giữa ảnh và CSV của URFD", "",
             f"Fall khớp hoàn toàn: {falls_ok}. ADL khớp: {adl_ok}. ADL `CSV_THIEU_FRAME`: {len(lacking)}. "
             "Không chuỗi nào bị loại, không cắt clip. Nguyên nhân chưa xác nhận.", "",
             "| Chuỗi | Số ảnh | Dòng CSV | Thiếu đầu | Thiếu giữa | Thiếu cuối |", "|---|---|---|---|---|---|"]
    for seq in lacking:
        lines.append(f"| {seq['seq']} | {seq['n_images']} | {seq['n_csv_rows']} | {seq['miss_head']} "
                     f"| {seq['gap_mid']} | {seq['miss_tail']} |")
    short = [f"{s['seq']} ({s['n_label1']})" for s in sequences
             if s["label"] == "fall" and s["n_label1"] < 15]
    lines += ["", f"Fall còn dưới 15 frame sau khi nằm ({len(short)} chuỗi, số frame trong ngoặc): "
              + ", ".join(short) + "."]
    return "\n".join(lines) + "\n"


def _le2i_section(meta: pd.DataFrame, bbox_stats: dict) -> str:
    odd = meta[(meta["dataset"] == "le2i") & (meta["has_event_label"] == 0)]["video_id"].tolist()
    return ("## Chỗ lệch và lưu ý của Le2i\n\n"
            f"- Video ODD (không có hai dòng đầu, `has_event_label = 0`, không vào `splits.json`): {', '.join(odd)}. "
            "Không phân biệt được té ngã hay chủ động nằm.\n"
            f"- Dòng bbox đã đọc: {bbox_stats['n_rows']}; bỏ dòng toàn 0: {bbox_stats['n_zero']}; "
            f"bỏ hộp không còn diện tích sau khi clip: {bbox_stats['n_degenerate']}.\n"
            "- bbox vượt khung là hợp lệ (người bị cắt ở mép), đã clip vào khung hình.\n")


def build_data_card(meta: pd.DataFrame, sequences: list[dict], bbox_stats: dict, sources: dict) -> str:
    """Ghép toàn bộ DATA_CARD.md. bbox_stats gồm n_rows, n_zero, n_degenerate (tính cả ODD)."""
    sections = ["# DATA CARD — nhom05-falldet-processed\n", _sources_section(sources), _counts_section(meta),
                _urfd_section(sequences), EXCLUSION, _le2i_section(meta, bbox_stats), CONVENTIONS, LICENSES]
    return "\n".join(sections)
