"""Bảng scene × kích thước khung × lớp của Le2i và khối cảnh báo "kích thước là đường tắt" cho DATA_CARD.

Đầu vào: metadata.csv (cột dataset, scene, scene_part, label, width, height). Đầu ra: DataFrame, Markdown.
Khối Markdown nằm giữa 2 dòng đánh dấu để EDA ghi đè kết quả đo độ tách mà không đụng phần còn lại của DATA_CARD.
Ví dụ: table = scene_size_table(meta); text = size_block(meta, separation=None).
"""
from __future__ import annotations

import pandas as pd

BLOCK_START = "<!-- size-risk:start -->"
BLOCK_END = "<!-- size-risk:end -->"
TABLE_COLUMNS = ["scene", "scene_part", "size", "fall", "adl", "total"]
DEFAULT_WARNING_THRESHOLD = 0.25  # silhouette theo kích thước từ mức này là có cấu trúc nhóm theo kích thước


def size_label(width, height) -> str:
    """Nhãn kích thước dạng '320×180'."""
    return f"{int(width)}×{int(height)}"


def scene_size_table(meta: pd.DataFrame, dataset: str = "le2i") -> pd.DataFrame:
    """Đếm video theo scene × phần × kích thước × lớp của 1 bộ dữ liệu (mặc định Le2i).

    Trả về cột scene, scene_part, size, fall, adl, total; chỉ tính video has_event_label = 1.
    """
    data = meta[(meta["dataset"] == dataset) & (meta["has_event_label"] == 1)].copy()
    data["scene_part"] = data["scene_part"].fillna("")
    data["size"] = [size_label(w, h) for w, h in zip(data["width"], data["height"])]
    counts = data.groupby(["scene", "scene_part", "size", "label"]).size().unstack("label", fill_value=0)
    for label in ("fall", "adl"):
        if label not in counts:
            counts[label] = 0
    table = counts[["fall", "adl"]].reset_index()
    table["total"] = table["fall"] + table["adl"]
    return table[TABLE_COLUMNS].sort_values(["scene", "scene_part", "size"]).reset_index(drop=True)


def single_class_sizes(table: pd.DataFrame) -> list[str]:
    """Các kích thước mà toàn bộ video thuộc đúng 1 lớp (mô hình có thể học kích thước thay cho tư thế)."""
    by_size = table.groupby("size")[["fall", "adl"]].sum()
    return [size for size, row in by_size.iterrows() if (row["fall"] == 0) != (row["adl"] == 0)]


def table_to_markdown(table: pd.DataFrame) -> str:
    """Bảng Markdown của scene_size_table."""
    lines = ["| Scene | Phần | Kích thước | Fall | ADL | Tổng |", "|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.scene} | {row.scene_part} | {row.size} | {row.fall} | {row.adl} | {row.total} |")
    return "\n".join(lines)


def static_warning(table: pd.DataFrame) -> str:
    """Câu cảnh báo tính từ bảng: nêu kích thước chỉ có 1 lớp, hoặc nói chưa thấy chênh lệch."""
    risky = single_class_sizes(table)
    if not risky:
        return "Mọi kích thước khung đều có cả fall và ADL trong Le2i."
    by_size = table.groupby("size")[["fall", "adl"]].sum()
    parts = []
    for size in risky:
        label = "fall" if by_size.loc[size, "adl"] == 0 else "ADL"
        parts.append(f"{int(by_size.loc[size].sum())} video {size}, tất cả đều là {label}")
    sizes_of_other = sorted(set(table["size"]) - set(risky))
    return (f"**Cảnh báo:** Le2i có {'; '.join(parts)}"
            + (f"; các video còn lại ({', '.join(sizes_of_other)}) có cả hai lớp" if sizes_of_other else "")
            + ". Nếu mô hình hoặc EDA dùng toạ độ pixel thì kích thước khung có thể trở thành đường tắt để phân biệt "
              "fall với ADL. Mọi đặc trưng phải tính theo tỉ lệ hoặc chuẩn hóa theo chiều cao người.")


def separation_text(separation: dict, threshold: float) -> str:
    """Đoạn Markdown mô tả kết quả đo độ tách theo kích thước trên đặc trưng cửa sổ."""
    silhouette = separation["silhouette_size"]
    within = separation.get("silhouette_size_fall")
    lines = [f"Độ tách theo kích thước trên {separation['n_windows']} cửa sổ ({', '.join(separation['sizes'])}): "
             f"silhouette = {silhouette:.3f}, khoảng cách centroid chuẩn hóa = {separation['centroid_distance']:.3f}"
             + (f"; riêng lớp fall (bỏ yếu tố lớp): silhouette = {within:.3f}" if within is not None else "")
             + f"; silhouette theo lớp để so sánh = {separation['silhouette_label']:.3f}."]
    worst = max(silhouette, within if within is not None else silhouette)
    if worst >= threshold:
        lines.append(f"**Cảnh báo: hai nhóm kích thước tách nhau rõ (silhouette ≥ {threshold:g}).** Kích thước khung có thể "
                     "là đường tắt; kiểm tra lại đặc trưng đang dùng toạ độ pixel và báo cáo kết quả riêng theo kích thước.")
    else:
        lines.append(f"Chưa thấy hai nhóm kích thước tách nhau (silhouette < {threshold:g}); cách đo chỉ là chỉ báo thô.")
    return "\n\n".join(lines)


def size_block(meta: pd.DataFrame, separation: dict | None = None, threshold: float = DEFAULT_WARNING_THRESHOLD) -> str:
    """Khối DATA_CARD 'Kích thước khung hình' có đánh dấu đầu/cuối; separation=None thì chưa có kết quả đo."""
    table = scene_size_table(meta)
    body = ["## Kích thước khung hình và nguy cơ đường tắt", "", "Le2i, video có nhãn, theo scene × kích thước × lớp "
            "(cũng lưu ở `reports/results/le2i_scene_size_table.csv` khi chạy EDA):", "", table_to_markdown(table), "",
            static_warning(table), ""]
    body.append(separation_text(separation, threshold) if separation
                else "Độ tách theo kích thước trên PCA/t-SNE: chưa đo, sẽ được ghi khi chạy `python -m src.eda.plots`.")
    return "\n".join([BLOCK_START, *body, BLOCK_END]) + "\n"


def replace_block(text: str, block: str) -> str:
    """Thay khối đánh dấu trong text bằng block; chưa có khối thì nối vào cuối. Chạy lại cho kết quả như nhau."""
    start, end = text.find(BLOCK_START), text.find(BLOCK_END)
    if start != -1 and end > start:
        return text[:start] + block.rstrip("\n") + text[end + len(BLOCK_END):]
    return text.rstrip("\n") + "\n\n" + block
