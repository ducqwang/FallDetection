"""Web Demo (Streamlit) — sản phẩm chung cả nhóm.

Cho người không chuyên: tải video/webcam -> hiển thị khung bao + ID ->
cảnh báo đỏ 'CẢNH BÁO TÉ NGÃ!'. Thanh trượt độ nhạy + chọn mô hình M1/M2/M3.
Chạy: streamlit run src/app.py
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

# Cho phép 'streamlit run src/app.py' từ gốc repo tìm thấy package 'src'.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="Fall Detection — Nhóm 05", page_icon="🚨", layout="wide")
st.title("🚨 Hệ thống phát hiện té ngã — Nhóm 05")

with st.sidebar:
    st.header("Cấu hình")
    model_choice = st.selectbox("Mô hình phân loại", ["M1 — Rule-based", "M2 — Pose+LSTM", "M3 — CNN+GRU"])
    sensitivity = st.slider("Độ nhạy (ngưỡng cảnh báo)", 0.0, 1.0, 0.5, 0.05)
    det_model = st.selectbox("Bộ phát hiện", ["yolo11n.pt", "yolo11s.pt"])
    source = st.radio("Nguồn", ["Tải video", "Webcam"])

uploaded = st.file_uploader("Chọn video (.mp4/.avi)", type=["mp4", "avi", "mov"]) if source == "Tải video" else None
status = st.empty()
alarm_box = st.empty()


def _run_inference(video_path: str):
    """Chạy detect+track + mô hình đã chọn -> danh sách frame cảnh báo + log."""
    from src.detection_tracking import DetectorTracker, write_mot
    from src.pipeline import alarms_rule, make_predict_fn, alarms_sequence_model, extract_id_crops
    from src.models.model1_rule import RuleThresholds

    dt = DetectorTracker(det_model)
    tracks = dt.run(video_path)
    out_dir = Path(tempfile.gettempdir()) / "fall_demo"
    out_dir.mkdir(exist_ok=True)
    write_mot(tracks, out_dir / "tracks_mot.txt")

    if model_choice.startswith("M1"):
        alarms = alarms_rule(tracks, RuleThresholds())
    else:
        # M2/M3: cần checkpoint đã huấn luyện (outputs/model2_best.pt | model3_best.pt)
        import torch
        if model_choice.startswith("M2"):
            from src.features import pose_keypoints
            from src.models.model2_pose_lstm import PoseLSTM
            kp = pose_keypoints(video_path)
            id_seq = {tid: (list(range(len(v))), v.reshape(len(v), -1)) for tid, v in kp.items()}
            model = PoseLSTM()
            model.load_state_dict(torch.load("outputs/model2_best.pt", map_location="cpu"))
        else:
            from src.features import cnn_embeddings
            from src.models.model3_cnn_gru import SeqGRU
            crops = extract_id_crops(video_path, tracks)
            id_seq = {tid: (fr, cnn_embeddings(c)) for tid, (fr, c) in crops.items()}
            model = SeqGRU()
            model.load_state_dict(torch.load("outputs/model3_best.pt", map_location="cpu"))
        alarms = alarms_sequence_model(id_seq, make_predict_fn(model), thresh=sensitivity)

    log = [{"frame": int(a)} for a in sorted(alarms)]
    (out_dir / "incidents.json").write_text(json.dumps(log, ensure_ascii=False, indent=2))
    return alarms, out_dir


if st.button("Bắt đầu phân tích"):
    if source == "Webcam":
        st.info("Webcam: dùng khi chạy local; trên Kaggle hãy tải video.")
    elif uploaded is None:
        st.warning("Hãy tải lên một video.")
    else:
        tmp = Path(tempfile.gettempdir()) / uploaded.name
        tmp.write_bytes(uploaded.read())
        with st.spinner("Đang phân tích..."):
            try:
                alarms, out_dir = _run_inference(str(tmp))
                if alarms:
                    alarm_box.error(f"🚨 CẢNH BÁO TÉ NGÃ! ({len(alarms)} sự kiện) — tại frame: {sorted(alarms)}")
                else:
                    status.success("Không phát hiện té ngã.")
                st.caption(f"Log sự cố + file MOT đã lưu tại: {out_dir}")
            except FileNotFoundError:
                st.error("Thiếu checkpoint mô hình (outputs/model2_best.pt / model3_best.pt). "
                         "Hãy huấn luyện M2/M3 trước, hoặc dùng M1 (không cần train).")
            except Exception as e:  # noqa: BLE001
                st.exception(e)
