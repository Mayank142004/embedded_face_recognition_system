"""
Face Recognition System — Admin Dashboard
Run with: streamlit run dashboard.py
"""

import os
import sys
import time
import threading
import importlib
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend Admin",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] .stMarkdown h2 { color: #a78bfa; font-size: 1.1rem; }

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-card .label { font-size: 0.85rem; color: rgba(255,255,255,0.55); margin-top: 0.2rem; }

/* Section headings */
.section-header {
    color: #e2d9f3;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(167,139,250,0.3);
}

/* Status badge */
.badge-running  { background:#10b981; color:#fff; padding:2px 10px; border-radius:999px; font-size:.78rem; }
.badge-stopped  { background:#ef4444; color:#fff; padding:2px 10px; border-radius:999px; font-size:.78rem; }

/* Buttons */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;  
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
div[data-testid="stButton"] button:hover { opacity: 0.85; transform: scale(1.02); }

/* Input */
.stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Log box */
.log-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #86efac;
    max-height: 220px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "📊 Dashboard",
        "running": False,
        "stop_flag": False,
        "frame_count": 0,
        "detected_count": 0,
        "attendance_today": 0,
        "training_log": [],
        "registered_employees": [],
        "live_source": "Webcam",
        # Recording state
        "recording": False,
        "recorded_video_path": None,
        "result_video_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────
# RECORDING DIRS
# ─────────────────────────────────────────────────────────
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
DATASET_DIR = "facenet_files/dataset2"
MODEL_PATH  = "facenet_models/new_classifier_Jun27_759.pkl"
ATTENDANCE_DIR = "marked_attendance"


def load_registered_employees():
    """Read employee list from the text file written by facent_svm_rec_passing.py."""
    txt = "registered_employees.txt"
    if os.path.exists(txt):
        with open(txt) as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback: list dataset sub-directories
    if os.path.isdir(DATASET_DIR):
        return sorted(os.listdir(DATASET_DIR))
    return []


def count_today_attendance():
    today = datetime.now().strftime('%Y_%m_%d')
    csv_path = os.path.join(ATTENDANCE_DIR, today, f"{today}_attendance_sheet.csv")
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path) as f:
        return max(0, sum(1 for _ in f) - 1)   # minus header row


def import_main_callback():
    """Import callback from main.py (models load once and are cached)."""
    if "main_module" not in st.session_state:
        import main as m
        st.session_state["main_module"] = m
    return st.session_state["main_module"].callback


# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 FaceAttend")
    st.caption("Admin Dashboard")
    st.markdown("---")

    pages = ["📊 Dashboard", "📷 Register Employee", "🔴 Live Analysis"]
    for p in pages:
        if st.button(p, key=f"nav_{p}",width='stretch'):
            st.session_state["page"] = p
            st.rerun()

    st.markdown("---")
    emps = load_registered_employees()
    st.markdown(f"**Registered Employees:** `{len(emps)}`")
    st.markdown(f"**Today's Attendance:** `{count_today_attendance()}`")
    st.markdown("---")
    st.caption("FaceRecognitionSystem v1.0")

page = st.session_state["page"]


# ═══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-header">📊 System Overview</p>', unsafe_allow_html=True)

    emps = load_registered_employees()
    today_att = count_today_attendance()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(emps)}</div><div class="label">Registered Employees</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">{today_att}</div><div class="label">Today\'s Attendance</div></div>', unsafe_allow_html=True)
    with c3:
        status_html = '<span class="badge-running">● Running</span>' if st.session_state["running"] else '<span class="badge-stopped">● Stopped</span>'
        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1rem">{status_html}</div><div class="label">Live Analysis Feed</div></div>', unsafe_allow_html=True)
    with c4:
        model_ok = os.path.exists(MODEL_PATH)
        model_ico = "✅" if model_ok else "❌"
        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.4rem">{model_ico}</div><div class="label">SVM Model Loaded</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_emp, col_att = st.columns(2)

    with col_emp:
        st.markdown("#### 👥 Registered Employees")
        if emps:
            for i, e in enumerate(emps, 1):
                st.write(f"`{i:02d}` {e}")
        else:
            st.info("No employees registered yet.")

    with col_att:
        st.markdown("#### 📅 Today's Attendance Log")
        today = datetime.now().strftime('%Y_%m_%d')
        csv_path = os.path.join(ATTENDANCE_DIR, today, f"{today}_attendance_sheet.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                st.dataframe(
                    [{k: v for k, v in r.items() if k != "Hyperlink"} for r in rows],
                    width="stretch"
                )
            else:
                st.info("No attendance records yet today.")
        else:
            st.info("No attendance records yet today.")


# ═══════════════════════════════════════════════════════════
# PAGE 2 — REGISTER EMPLOYEE
# ═══════════════════════════════════════════════════════════
elif page == "📷 Register Employee":
    st.markdown('<p class="section-header">📷 Register New Employee</p>', unsafe_allow_html=True)

    st.markdown("""
    **Steps:**
    1. Enter the employee's full name.
    2. Capture **5–15 photos** using the camera button.
    3. Click **Save Images** to store them in the dataset.
    4. Click **Train Model** to update the classifier.
    """)
    st.markdown("---")

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        emp_name = st.text_input("👤 Employee Name", placeholder="e.g. John Doe")
        emp_name = emp_name.strip()

        cam_img = st.camera_input("📸 Capture Face Photo")

        if st.button("💾 Save This Image", disabled=(not emp_name or cam_img is None)):
            if emp_name and cam_img:
                save_dir = os.path.join(DATASET_DIR, emp_name)
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(save_dir, f"{ts}.jpg")
                # Convert streamlit UploadedFile bytes → numpy → save
                file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                cv2.imwrite(filename, img)
                st.success(f"✅ Image saved: `{filename}`")

    with col_preview:
        st.markdown("#### 📁 Saved Images for This Employee")
        if emp_name:
            emp_dir = os.path.join(DATASET_DIR, emp_name)
            if os.path.isdir(emp_dir):
                imgs = [f for f in os.listdir(emp_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                st.write(f"**{len(imgs)} image(s)** saved for `{emp_name}`")
                if imgs:
                    # show the last 6 in a grid
                    cols = st.columns(3)
                    for i, fname in enumerate(sorted(imgs)[-6:]):
                        with cols[i % 3]:
                            fpath = os.path.join(emp_dir, fname)
                            img = cv2.imread(fpath)
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, width=200)
            else:
                st.info("No images saved yet. Use the camera button to capture some.")
        else:
            st.info("Enter employee name to see saved images.")

    st.markdown("---")
    st.markdown("### 🧠 Retrain Model")
    st.warning("⚠️ Training may take several minutes. Do not close this tab.")

    if st.button("🚀 Train Model with Updated Dataset"):
        log_placeholder = st.empty()
        result_placeholder = st.empty()
        log_lines = []

        def append_log(msg):
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            log_html = "<br>".join(log_lines[-20:])
            log_placeholder.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

        try:
            # We import training here so the heavy imports only happen when needed
            import training
            importlib.reload(training)  # pick up fresh state
            append_log("Starting training pipeline...")
            result = training.train_model(status_callback=append_log)
            result_placeholder.success(
                f"🎉 Training complete!\n\n"
                f"**Train accuracy:** {result['train_acc']:.2%}  |  "
                f"**Test accuracy:** {result['test_acc']:.2%}\n\n"
                f"**Registered employees:** {', '.join(result['classes'])}"
            )
            # Force reload of the inference module so main.py picks up the new model
            if "main_module" in st.session_state:
                del st.session_state["main_module"]
            from facenet_files.facent_svm_rec_passing import load_model
            load_model()
        except Exception as e:
            st.error(f"❌ Training failed: {e}")


# ═══════════════════════════════════════════════════════════
# PAGE 3 — LIVE ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "🔴 Live Analysis":
    st.markdown('<p class="section-header">🔴 Live Face Recognition Analysis</p>', unsafe_allow_html=True)

    ctrl_col, info_col = st.columns([1, 2], gap="large")

    with ctrl_col:
        source = st.selectbox(
            "📹 Video Source",
            ["Webcam — Live View", "Webcam — Record & Analyze", "Upload Video File"],
            key="live_source_select",
        )

        camera_index = 0
        if "Webcam" in source:
            camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1, help="If the camera fails to open, try changing this to 1 or 2.")

        uploaded_video = None
        if source == "Upload Video File":
            uploaded_video = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov"])

        st.markdown("---")

        is_running   = st.session_state["running"]
        is_recording = st.session_state["recording"]

        # ── Webcam live view controls ─────────────────────
        if source == "Webcam — Live View":
            start_btn = st.button("▶ Start Analysis",  disabled=is_running,  key="btn_start_live")
            stop_btn  = st.button("⏹ Stop Analysis",   disabled=not is_running, key="btn_stop_live")
            record_start_btn = record_stop_btn = analyze_recorded_btn = False

        # ── Record & Analyze controls ─────────────────────
        elif source == "Webcam — Record & Analyze":
            record_start_btn = st.button("⏺ Start Recording", disabled=is_recording, key="btn_rec_start")
            record_stop_btn  = st.button("⏹ Stop Recording",  disabled=not is_recording, key="btn_rec_stop")
            analyze_recorded_btn = False
            start_btn = stop_btn = False

            # Show previous recording if any
            if st.session_state.get("recorded_video_path") and not is_recording:
                rec_path = st.session_state["recorded_video_path"]
                if os.path.exists(rec_path):
                    st.info(f"Last recording: `{rec_path}`")
                    analyze_recorded_btn = st.button(
                        "🔍 Analyze Recorded Video",
                        key="btn_analyze_rec",
                        disabled=is_running,
                    )

        # ── Upload video controls ─────────────────────────
        else:
            start_btn = st.button("▶ Analyze Video", disabled=is_running, key="btn_start_upload")
            stop_btn  = st.button("⏹ Stop",          disabled=not is_running, key="btn_stop_upload")
            record_start_btn = record_stop_btn = analyze_recorded_btn = False

        st.markdown("---")
        frames_metric   = st.empty()
        detected_metric = st.empty()

    with info_col:
        feed_label = st.empty()
        feed_label.markdown("##### 🎥 Live Feeds")
        
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            st.markdown("**Input (Raw)**")
            raw_frame_placeholder = st.empty()
        with feed_col2:
            st.markdown("**Output (Analyzed)**")
            analyzed_frame_placeholder = st.empty()
            
        st.markdown("##### 📋 Recognition Events")
        events_placeholder = st.empty()

        # ── Post-session video playback area ─────────────
        playback_area = st.empty()

    # ════════════════════════════════════════════════════
    # STOP BUTTON HANDLERS  (must be before start blocks)
    # ════════════════════════════════════════════════════
    if stop_btn:
        st.session_state["stop_flag"] = True
        st.session_state["running"]   = False
        st.rerun()

    if record_stop_btn:
        st.session_state["stop_flag"] = True
        st.session_state["recording"] = False
        st.rerun()

    # ════════════════════════════════════════════════════
    # HELPER — show recorded + result videos side by side
    # ════════════════════════════════════════════════════
    def show_session_videos():
        rec_path    = st.session_state.get("recorded_video_path")
        result_path = st.session_state.get("result_video_path")
        has_rec     = rec_path    and os.path.exists(rec_path)
        has_res     = result_path and os.path.exists(result_path)
        if has_rec or has_res:
            with playback_area.container():
                st.markdown("---")
                st.markdown("### 🎬 Session Videos")
                v1, v2 = st.columns(2)
                if has_rec:
                    with v1:
                        st.markdown("**📹 Captured / Input Video**")
                        with open(rec_path, "rb") as f:
                            st.video(f.read())
                        st.caption(rec_path)
                if has_res:
                    with v2:
                        st.markdown("**🔍 Analyzed Result Video**")
                        with open(result_path, "rb") as f:
                            st.video(f.read())
                        st.caption(result_path)

    # Show previous session videos immediately on page load
    show_session_videos()

    # ════════════════════════════════════════════════════
    # HELPER — run the frame processing loop
    # ════════════════════════════════════════════════════
    def run_analysis_loop(
        cap,
        record_raw: bool = False,
        raw_writer=None,
        result_writer=None,
        target_fps: int = 30,
        ui_update_every: int = 2,
    ):
        """
        Shared frame loop used by all three analysis modes.

        Args:
            cap             — open cv2.VideoCapture
            record_raw      — whether to write raw frames to raw_writer
            raw_writer      — cv2.VideoWriter for the input recording (can be None)
            result_writer   — cv2.VideoWriter for the annotated result (can be None)
            target_fps      — playback FPS for saved video files (always 30)
            ui_update_every — only push every Nth frame to st.image() to reduce
                              websocket churn; every frame is still written to file

        Frame-pacing strategy (Task 2):
            Model inference (YOLO + FaceNet + SVM) is slower than 30 fps.
            Without pacing, simply stamping every processed frame at 33ms intervals
            causes the saved video to play back at fast-forward speed.
            Solution: after computing each annotated frame, enter a while loop that
            writes the *most recently processed* frame to result_writer as many times
            as needed to fill the wall-clock gap at the target_fps cadence.

        UI throttle strategy (Task 3):
            st.image() re-renders the full frame over the websocket on every call;
            pushing every frame at inference speed saturates the connection and causes
            jank. Mitigation:
              1. Only call st.image() when idx % ui_update_every == 0.
              2. Downscale the preview frame to max 480px wide before pushing.
              3. Throttle with time.time() so UI updates target ~12 fps cadence.

        NOTE (stretch goal): streamlit-webrtc is the architecturally correct solution
        for true real-time smooth webcam preview. It uses a proper media stream
        pipeline (WebRTC / RTP) instead of per-frame image re-renders over the
        Streamlit websocket, which would eliminate the jank entirely. Consider
        migrating the "Webcam — Live View" mode to streamlit-webrtc in a future PR.
        """
        callback_fn = import_main_callback()
        events = []
        idx    = 0
        import csv as csv_mod

        # ── Frame-pacing state ────────────────────────────────────────────────
        frame_interval  = 1.0 / target_fps
        next_write_time = time.time()

        # ── UI throttle state (Task 3) ────────────────────────────────────────
        ui_frame_interval = 1.0 / 12.0   # target ~12 fps for Streamlit preview
        last_ui_update    = 0.0           # wall-clock time of last st.image() call

        # Preview downscale width (px) — full resolution still written to file
        PREVIEW_WIDTH = 480

        def _downscale_for_preview(bgr_frame: np.ndarray) -> np.ndarray:
            h, w = bgr_frame.shape[:2]
            if w <= PREVIEW_WIDTH:
                return bgr_frame
            scale = PREVIEW_WIDTH / w
            return cv2.resize(bgr_frame, (PREVIEW_WIDTH, int(h * scale)),
                              interpolation=cv2.INTER_AREA)

        last_annotated = None   # most recently processed annotated frame

        while not st.session_state["stop_flag"]:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Write raw frame to file (not paced — we write every captured frame)
            if record_raw and raw_writer:
                raw_writer.write(frame)

            # ── Run recognition ───────────────────────────────────────────────
            try:
                annotated = callback_fn(frame, idx)
            except Exception as e:
                annotated = frame.copy()
                cv2.putText(annotated, f"Err: {e}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            last_annotated = annotated

            # ── Frame-pacing write to result file (Task 2) ───────────────────
            # Duplicate the latest annotated frame to fill wall-clock gaps so
            # the saved video plays back at correct real-time speed.
            if result_writer:
                now = time.time()
                while next_write_time <= now:
                    result_writer.write(last_annotated)
                    next_write_time += frame_interval

            # ── UI update with throttle + downscale (Task 3) ─────────────────
            now = time.time()
            push_to_ui = (
                idx % ui_update_every == 0
                and (now - last_ui_update) >= ui_frame_interval
            )
            if push_to_ui:
                preview_raw      = _downscale_for_preview(frame)
                preview_analyzed = _downscale_for_preview(annotated)

                raw_frame_placeholder.image(
                    cv2.cvtColor(preview_raw, cv2.COLOR_BGR2RGB),
                    channels="RGB", width="stretch",
                )
                analyzed_frame_placeholder.image(
                    cv2.cvtColor(preview_analyzed, cv2.COLOR_BGR2RGB),
                    channels="RGB", width="stretch",
                )
                last_ui_update = now

            idx += 1
            st.session_state["frame_count"] = idx
            frames_metric.metric("🖼️ Frames Processed", idx)

            # ── Attendance events (read CSV written by callback) ──────────────
            today    = datetime.now().strftime('%Y_%m_%d')
            csv_path = os.path.join(ATTENDANCE_DIR, today, f"{today}_attendance_sheet.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    reader = csv_mod.DictReader(f)
                    events = list(reader)
                events_placeholder.dataframe(
                    [{k: v for k, v in r.items() if k != "Hyperlink"} for r in events[-10:]],
                    width="stretch",
                )
                detected_metric.metric("✅ Logged Today", len(events))

        return idx


    # ════════════════════════════════════════════════════
    # MODE 1 — Webcam Live View
    # ════════════════════════════════════════════════════
    if start_btn and source == "Webcam — Live View":
        st.session_state.update(running=True, stop_flag=False,
                                frame_count=0, detected_count=0,
                                result_video_path=None, recorded_video_path=None)

        with st.spinner("Loading models…"):
            import_main_callback()



        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not cap.isOpened():
            st.error(
                f"❌ Cannot open webcam (Index {camera_index}). "
                "Make sure no other application is using the camera."
            )
            st.session_state["running"] = False
            st.stop()

        try:
            TARGET_FPS = 30   # always write at 30 fps regardless of webcam's reported fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(RECORDINGS_DIR, f"result_{ts}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            result_writer = cv2.VideoWriter(result_path, fourcc, TARGET_FPS, (w, h))

            n = run_analysis_loop(
                cap,
                record_raw=False,
                result_writer=result_writer,
                target_fps=TARGET_FPS,
            )

            st.session_state["result_video_path"] = result_path
            st.success(f"✅ Analysis complete. {n} frames processed.")

        finally:
            cap.release()

            if 'result_writer' in locals():
                result_writer.release()

            st.session_state["running"] = False
            st.session_state["stop_flag"] = False



    # ════════════════════════════════════════════════════
    # MODE 2a — Record Webcam
    # ════════════════════════════════════════════════════
    if record_start_btn and source == "Webcam — Record & Analyze":
        st.session_state.update(recording=True, stop_flag=False,
                                frame_count=0, detected_count=0,
                                recorded_video_path=None, result_video_path=None)

        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():  
            st.error(f"❌ Cannot open webcam (Index {camera_index}). Try changing the Camera Index.")
            st.session_state["recording"] = False
            st.stop()

        TARGET_FPS = 30   # always write at 30 fps regardless of webcam's reported fps
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path    = os.path.join(RECORDINGS_DIR, f"recorded_{ts}.mp4")
        result_path = os.path.join(RECORDINGS_DIR, f"result_{ts}.mp4")

        fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
        raw_writer    = cv2.VideoWriter(raw_path,    fourcc, TARGET_FPS, (w, h))
        result_writer = cv2.VideoWriter(result_path, fourcc, TARGET_FPS, (w, h))

        feed_label.markdown("##### 🎥 Live Feed  <span style='color:#ef4444'>⏺ Recording</span>",
                            unsafe_allow_html=True)

        with st.spinner("Loading models…"):
            import_main_callback()

        n = run_analysis_loop(cap, record_raw=True, raw_writer=raw_writer,
                              result_writer=result_writer, target_fps=TARGET_FPS)

        cap.release()
        raw_writer.release()
        result_writer.release()
        st.session_state["recording"] = False
        st.session_state["recorded_video_path"] = raw_path
        st.session_state["result_video_path"]   = result_path
        st.success(f"✅ Recording stopped. {n} frames saved.")
        feed_label.markdown("##### 🎥 Live Feed")
        show_session_videos()

    # ════════════════════════════════════════════════════
    # MODE 2b — Analyze previously recorded video
    # ════════════════════════════════════════════════════
    if analyze_recorded_btn:
        rec_path = st.session_state["recorded_video_path"]
        st.session_state.update(running=True, stop_flag=False,
                                frame_count=0, result_video_path=None)

        cap = cv2.VideoCapture(rec_path)
        if not cap.isOpened():
            st.error(f"❌ Cannot open recorded file: {rec_path}")
            st.session_state["running"] = False
            st.stop()

        TARGET_FPS    = 30   # always write at 30 fps
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path   = os.path.join(RECORDINGS_DIR, f"result_{ts}.mp4")
        fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
        result_writer = cv2.VideoWriter(result_path, fourcc, TARGET_FPS, (w, h))

        with st.spinner("Loading models…"):
            import_main_callback()

        n = run_analysis_loop(cap, record_raw=False, result_writer=result_writer,
                              target_fps=TARGET_FPS)

        cap.release()
        result_writer.release()
        st.session_state["running"]          = False
        st.session_state["result_video_path"] = result_path
        st.success(f"✅ Analysis of recorded video complete. {n} frames processed.")
        show_session_videos()

    # ════════════════════════════════════════════════════
    # MODE 3 — Uploaded Video File
    # ════════════════════════════════════════════════════
    if start_btn and source == "Upload Video File":
        if uploaded_video is None:
            st.error("Please upload a video file first.")
            st.stop()

        st.session_state.update(running=True, stop_flag=False,
                                frame_count=0, result_video_path=None,
                                recorded_video_path=None)

        # Save upload to a temp path
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path  = os.path.join(RECORDINGS_DIR, f"uploaded_{ts}.mp4")
        with open(raw_path, "wb") as f:
            f.write(uploaded_video.read())
        st.session_state["recorded_video_path"] = raw_path

        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            st.error("❌ Cannot open uploaded file.")
            st.session_state["running"] = False
            st.stop()

        TARGET_FPS    = 30   # always write at 30 fps
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result_path   = os.path.join(RECORDINGS_DIR, f"result_{ts}.mp4")
        fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
        result_writer = cv2.VideoWriter(result_path, fourcc, TARGET_FPS, (w, h))

        with st.spinner("Loading models…"):
            import_main_callback()

        n = run_analysis_loop(cap, record_raw=False, result_writer=result_writer,
                              target_fps=TARGET_FPS)

        cap.release()
        result_writer.release()
        st.session_state["running"]          = False
        st.session_state["result_video_path"] = result_path
        st.info(f"✅ Video analysis complete. {n} frames processed.")
        show_session_videos()

