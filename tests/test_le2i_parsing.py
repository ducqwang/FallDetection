"""Test đọc Le2i: ghép annotation, định dạng header_top và header_mid, chuẩn hóa bbox, dòng metadata.

Đầu vào: file nhỏ tự tạo trong tmp_path. Kiểm tra: cặp ở đầu, giữa, cuối file và mọi trường hợp phải raise.
Ví dụ: python -m pytest tests/test_le2i_parsing.py -q
"""
import numpy as np
import pytest

from src.data import le2i
from src.data.common import GT_BBOX_COLUMNS, DataCheckError
from src.data.metadata import le2i_row, metadata_frame


def le2i_tree(root):
    for folder, annot, number, text in (("Home_01", "Annotation_files", 1, "1\n2\n1,1,1,1,5,5\n"),
                                        ("Coffee_room_02", "Annotations_files", 1, "0\n0\n1,1,1,1,5,5\n")):
        videos = root / folder / folder / "Videos"
        notes = root / folder / folder / annot
        videos.mkdir(parents=True)
        notes.mkdir(parents=True)
        (videos / f"video ({number}).avi").touch()
        (notes / f"video ({number}).txt").write_text(text)
    office = root / "Office" / "Office"
    office.mkdir(parents=True)
    (office / "video (1).avi").touch()


def test_le2i_same_number_different_scene_and_annotation_folder_names(tmp_path):
    le2i_tree(tmp_path)
    assert le2i.find_le2i_root(tmp_path.parent) == tmp_path
    videos = {v["key"]: v for v in le2i.scan_videos(tmp_path)}
    assert set(videos) == {"home|01|1", "coffee_room|02|1", "office||1"}
    assert videos["home|01|1"]["annotation"].parent.name == "Annotation_files"
    assert videos["coffee_room|02|1"]["annotation"].parent.name == "Annotations_files"
    assert videos["office||1"]["annotation"] is None


def annotation(tmp_path, text):
    path = tmp_path / "video (1).txt"
    path.write_text(text)
    return le2i.parse_annotation(path)


def bbox_lines(first, last):
    return [f"{frame},1,10,20,60,120" for frame in range(first, last + 1)]


def test_parse_fall_header_keeps_aux_code(tmp_path):
    parsed = annotation(tmp_path, "48\n80\n1,7,272,48,312,207\n")
    assert (parsed["kind"], parsed["start"], parsed["end"]) == ("fall", 48, 80)
    assert parsed["label_source"] == "header_top" and parsed["header_lines"] == (1, 2)
    assert parsed["boxes"].tolist() == [[1, 7, 272, 48, 312, 207]]


def test_parse_adl_header(tmp_path):
    parsed = annotation(tmp_path, "0\n0\n1,1,1,1,5,5\n")
    assert parsed["kind"] == "adl" and parsed["label_source"] == "header_top"


@pytest.mark.parametrize("start,end,first_line,last_frame", [(197, 227, 219, 278), (1816, 1852, 1894, 1954), (87, 113, 156, 215)])
def test_pair_in_middle_like_the_three_real_files(tmp_path, start, end, first_line, last_frame):
    lines = bbox_lines(1, first_line - 1) + [str(start), str(end)] + bbox_lines(first_line, last_frame)
    parsed = annotation(tmp_path, "\n".join(lines))
    assert (parsed["kind"], parsed["start"], parsed["end"]) == ("fall", start, end)
    assert parsed["label_source"] == "header_mid" and parsed["header_lines"] == (first_line, first_line + 1)
    assert parsed["last_frame"] == last_frame and len(parsed["boxes"]) == last_frame
    assert parsed["boxes"][:, 0].tolist() == list(range(1, last_frame + 1))


def test_pair_at_end_of_file_is_valid(tmp_path):
    parsed = annotation(tmp_path, "\n".join(bbox_lines(1, 20) + ["5", "15"]))
    assert parsed["label_source"] == "header_mid" and (parsed["start"], parsed["end"]) == (5, 15)
    assert parsed["header_lines"] == (21, 22)


def test_end_may_equal_last_frame(tmp_path):
    assert annotation(tmp_path, "\n".join(bbox_lines(1, 20) + ["5", "20"]))["end"] == 20


@pytest.mark.parametrize("name,lines", [
    ("không có cặp", bbox_lines(1, 20)),
    ("một dòng đơn-số", bbox_lines(1, 10) + ["5"] + bbox_lines(11, 20)),
    ("hai dòng đơn-số không liền nhau", bbox_lines(1, 5) + ["5"] + bbox_lines(6, 10) + ["9"] + bbox_lines(11, 20)),
    ("ba dòng đơn-số liền nhau", bbox_lines(1, 10) + ["5", "9", "12"] + bbox_lines(11, 20)),
    ("hai cặp", bbox_lines(1, 5) + ["2", "4"] + bbox_lines(6, 10) + ["7", "9"] + bbox_lines(11, 20)),
    ("start bằng end", bbox_lines(1, 10) + ["5", "5"] + bbox_lines(11, 20)),
    ("start lớn hơn end", bbox_lines(1, 10) + ["9", "5"] + bbox_lines(11, 20)),
    ("start bằng 0", bbox_lines(1, 10) + ["0", "5"] + bbox_lines(11, 20)),
    ("end vượt frame lớn nhất", bbox_lines(1, 10) + ["5", "21"] + bbox_lines(11, 20)),
    ("frame có lỗ", bbox_lines(1, 10) + ["5", "9"] + bbox_lines(12, 20)),
    ("frame lặp", bbox_lines(1, 10) + ["5", "9"] + bbox_lines(10, 20)),
    ("frame không bắt đầu từ 1", bbox_lines(2, 10) + ["5", "9"] + bbox_lines(11, 20)),
    ("cặp có dòng lạ khác", bbox_lines(1, 10) + ["5", "9"] + ["abc"] + bbox_lines(11, 20)),
])
def test_other_formats_raise_with_file_name(tmp_path, name, lines):
    with pytest.raises(DataCheckError, match="video"):
        annotation(tmp_path, "\n".join(lines))


@pytest.mark.parametrize("stray", ["197", "1,1,5", "a,b,c,d,e,f"])
def test_valid_header_with_stray_line_raises(tmp_path, stray):
    with pytest.raises(DataCheckError, match="video"):
        annotation(tmp_path, "\n".join(["48", "80"] + bbox_lines(1, 5) + [stray] + bbox_lines(6, 10)))


def test_parse_rejects_reversed_interval(tmp_path):
    with pytest.raises(DataCheckError):
        annotation(tmp_path, "50\n10\n1,1,1,1,5,5\n")


def test_clean_boxes_zero_rows_clip_and_zero_based():
    boxes = np.array([[1, 1, 0, 0, 0, 0], [2, 3, -5, 10, 330, 250], [3, 1, 325, 10, 340, 50]], dtype=float)
    table, stats = le2i.clean_boxes("v", boxes, 320, 240, 10)
    assert stats == {"n_rows": 3, "n_zero": 1, "n_degenerate": 1}
    assert table.iloc[0][["frame_idx", "x1", "y1", "x2", "y2", "aux_code"]].tolist() == [1, 0, 10, 320, 240, 3]
    assert str(table["aux_code"].dtype).startswith("int") and list(table.columns) == GT_BBOX_COLUMNS


def test_clean_boxes_rejects_frame_outside_video():
    with pytest.raises(DataCheckError):
        le2i.clean_boxes("v", np.array([[11, 1, 1, 1, 5, 5]], dtype=float), 320, 240, 10)


def test_le2i_row_fall_header_top_and_header_mid_are_zero_based():
    base = {"scene": "home", "part": "02", "number": 31, "fps": 24.0, "n_frames": 240, "width": 320, "height": 180}
    top = le2i_row({**base, "kind": "fall", "start": 48, "end": 80, "label_source": "header_top"})
    mid = le2i_row({**base, "kind": "fall", "start": 197, "end": 227, "label_source": "header_mid"})
    assert top["video_id"] == "le2i_home02_031" and top["fall_start_frame"] == 47 and top["fall_end_frame"] == 79
    assert top["duration_s"] == 10.0 and top["has_event_label"] == 1 and top["label_source"] == "header_top"
    assert (mid["fall_start_frame"], mid["fall_end_frame"]) == (196, 226)
    assert mid["label"] == "fall" and mid["has_event_label"] == 1 and mid["label_source"] == "header_mid"


def test_le2i_row_without_annotation_has_no_event_label():
    base = {"scene": "home", "part": "02", "number": 31, "fps": 24.0, "n_frames": 240, "width": 320, "height": 180}
    row = le2i_row({**base, "kind": "no_annot", "start": None, "end": None, "label_source": ""})
    assert row["has_event_label"] == 0 and row["label"] == "" and row["fall_start_frame"] is None


def test_metadata_frame_keeps_integers_and_rejects_duplicates():
    base = {"scene": "home", "part": "02", "number": 1, "fps": 24.0, "n_frames": 240, "width": 320, "height": 180}
    row = le2i_row({**base, "kind": "fall", "start": 48, "end": 80, "label_source": "header_top"})
    assert str(metadata_frame([row])["fall_start_frame"].dtype) == "Int64"
    with pytest.raises(ValueError):
        metadata_frame([row, row])
