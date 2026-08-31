"""
Face Recognition System — Admin Dashboard (MongoDB + GCS Version)
Run with: streamlit run dashboard.py
"""
import os
import time
import importlib
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import pandas as pd

from config import RECORDINGS_DIR, DATASET_DIR
from db import (
    get_all_employees,
    get_today_attendance,
    get_today_attendance_count,
    get_today_local_attendance,
    register_employee,
)
from gcs_storage import upload_employee_photo, upload_embedding
from facenet_files.facent_svm_rec_passing import get_embedding

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAttend Admin",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
/* Sidebar */
[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); backdrop-filter: blur(20px); }
[data-testid="stSidebar"] .stMarkdown h2 { color: #a78bfa; font-size: 1.1rem; }
/* Cards */
.metric-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; padding: 1.2rem 1.5rem; backdrop-filter: blur(12px); margin-bottom: 1rem;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-card .label { font-size: 0.85rem; color: rgba(255,255,255,0.55); margin-top: 0.2rem; }
.section-header { color: #e2d9f3; font-size: 1.4rem; font-weight: 600; margin-bottom: 0.5rem; padding-bottom: 0.4rem; border-bottom: 2px solid rgba(167,139,250,0.3); }
.badge-running  { background:#10b981; color:#fff; padding:2px 10px; border-radius:999px; font-size:.78rem; }
.badge-stopped  { background:#ef4444; color:#fff; padding:2px 10px; border-radius:999px; font-size:.78rem; }
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; border: none;  
    border-radius: 10px; font-weight: 600; padding: 0.5rem 1.5rem;
}
.stTextInput input, .stSelectbox select { background: rgba(255,255,255,0.06) !important; color: white !important; }
.log-box { background: rgba(0,0,0,0.4); padding: 1rem; font-family: monospace; color: #86efac; max-height: 220px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# STATE & HELPERS
# ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "📊 Dashboard",
        "running": False,
        "stop_flag": False,
        "frame_count": 0,
        "live_source": "Webcam",
        "recording": False,
        "recorded_video_path": None,
        "result_video_path": None,
        "registration_photos": [],
        "last_cam_hash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


def import_main_callback():
    if "main_module" not in st.session_state:
        import main as m
        st.session_state["main_module"] = m
    return st.session_state["main_module"].callback


def import_local_callback():
    """Load main_local.py models and return callback_local. Isolated from Pi mode."""
    if "local_module" not in st.session_state:
        import main_local as ml
        ml.load_local_models()
        st.session_state["local_module"] = ml
    else:
        # Ensure models are loaded even if module was already imported
        st.session_state["local_module"].load_local_models()
    # Reset per-session state so IDs don't carry over from previous runs
    st.session_state["local_module"].reset_local_state()
    return st.session_state["local_module"].callback_local


emps = get_all_employees()
today_att = get_today_attendance_count()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 FaceAttend")
    st.caption("Admin Dashboard")
    st.markdown("---")
    pages = ["📊 Dashboard", "📷 Register Employee", "🔴 Live Analysis"]
    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state["page"] = p
            st.rerun()

    st.markdown("---")
    st.markdown(f"**Registered Employees:** `{len(emps)}`")
    st.markdown(f"**Today's Attendance:** `{today_att}`")
    st.markdown("---")
    st.caption("FaceRecognitionSystem v2.0")

page = st.session_state["page"]


# ═══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<p class="section-header">📊 System Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(emps)}</div><div class="label">Registered Employees</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="value">{today_att}</div><div class="label">Today\'s Attendance</div></div>', unsafe_allow_html=True)
    with c3:
        status_html = '<span class="badge-running">● Running</span>' if st.session_state["running"] else '<span class="badge-stopped">● Stopped</span>'
        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1rem">{status_html}</div><div class="label">Live Analysis Feed</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.4rem">☁️</div><div class="label">GCS / MongoDB Sync</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_emp, col_att = st.columns(2)

    with col_emp:
        st.markdown("#### 👥 Registered Employees")
        if emps:
            df_emps = pd.DataFrame(emps)[["emp_id", "emp_name", "registered_at"]]
            st.dataframe(df_emps, use_container_width=True)
        else:
            st.info("No employees registered yet.")

    with col_att:
        st.markdown("#### 📅 Today's Attendance Log")
        att_logs = get_today_attendance()
        if att_logs:
            df_att = pd.DataFrame(att_logs)[["emp_id", "emp_name", "timestamp", "status"]]
            st.dataframe(df_att, use_container_width=True)
        else:
            st.info("No attendance records yet today.")


# ═══════════════════════════════════════════════════════════
# PAGE 2 — REGISTER EMPLOYEE
# ═══════════════════════════════════════════════════════════
elif page == "📷 Register Employee":
    st.markdown('<p class="section-header">📷 Register New Employee</p>', unsafe_allow_html=True)

    col_form, col_preview = st.columns([1, 1], gap="large")

    with col_form:
        emp_id = st.text_input("🆔 Employee ID (Alphanumeric)", placeholder="e.g. E001").strip()
        emp_name = st.text_input("👤 Employee Name", placeholder="e.g. John Doe").strip()

        st.info("Take 5 to 15 different photos. Move your head slightly for better accuracy.")
        cam_img = st.camera_input("📸 Capture Face Photo")
        
        # Auto-add new photo to collection
        if cam_img is not None:
            import hashlib
            cam_bytes = cam_img.getvalue()
            cam_hash = hashlib.md5(cam_bytes).hexdigest()
            if st.session_state.get("last_cam_hash") != cam_hash:
                st.session_state["last_cam_hash"] = cam_hash
                file_bytes = np.asarray(bytearray(cam_bytes), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state["registration_photos"].append(img)
                st.rerun()

    with col_preview:
        num_photos = len(st.session_state["registration_photos"])
        st.markdown(f"#### 📁 Collected Photos: {num_photos} (Minimum: 5)")
        
        if num_photos > 0:
            cols = st.columns(3)
            for idx, img in enumerate(st.session_state["registration_photos"]):
                with cols[idx % 3]:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                    if st.button("❌ Remove", key=f"del_{idx}"):
                        st.session_state["registration_photos"].pop(idx)
                        st.rerun()
        else:
            st.info("Your collected photos will appear here.")

    st.markdown("---")
    
    # ── Finalize Registration ──
    can_register = bool(emp_id and emp_name and num_photos >= 5)
    
    if st.button("💾 Finalize & Register Employee", disabled=not can_register):
        save_dir = os.path.join(DATASET_DIR, emp_id)
        os.makedirs(save_dir, exist_ok=True)
        
        embeddings_list = []
        uploaded_gcs_url = None
        local_paths_saved = []
        
        with st.spinner(f"Processing {num_photos} photos and generating master embedding..."):
            try:
                for idx, img in enumerate(st.session_state["registration_photos"]):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{ts}_{idx}.jpg"
                    local_path = os.path.join(save_dir, filename)
                    
                    # 1. Save locally
                    cv2.imwrite(local_path, img)
                    local_paths_saved.append(local_path)
                    
                    # 2. Extract Embedding
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_img = cv2.resize(img_rgb, (160, 160))
                    emb = get_embedding(face_img)
                    embeddings_list.append(emb)
                    
                    # 3. Best-effort upload to GCS (save the first successful URL)
                    try:
                        gcs_url = upload_employee_photo(emp_id, local_path, filename)
                        if not uploaded_gcs_url:
                            uploaded_gcs_url = gcs_url
                    except Exception:
                        pass # Ignore individual upload failures
                
                # 4. Calculate master embedding (mathematical average of all photos)
                master_embedding = np.mean(embeddings_list, axis=0)
                
                # 5. Upload master embedding to GCS
                emb_gcs = None
                try:
                    emb_gcs = upload_embedding(emp_id, master_embedding)
                except Exception:
                    pass

                # 6. Save to MongoDB
                register_employee(
                    emp_id=emp_id,
                    emp_name=emp_name,
                    photo_local_path=local_paths_saved[0],
                    photo_gcs_url=uploaded_gcs_url,
                    face_embedding=master_embedding.tolist()
                )
                
                # 7. Clear the session state for the next person
                st.session_state["registration_photos"] = []
                st.session_state["last_cam_hash"] = None
                
                if uploaded_gcs_url and emb_gcs:
                    st.success(f"✅ Registered {emp_name} ({emp_id}) successfully!\nMaster embedding and {num_photos} photos synced to GCS.")
                else:
                    st.warning(f"✅ Registered {emp_name} ({emp_id}) successfully!\n☁️ GCP upload failed. Master embedding and {num_photos} photos saved locally to: `{save_dir}`")
                    
            except Exception as e:
                st.error(f"❌ Registration error: {e}")

    st.markdown("---")
    st.markdown("### 🧠 Retrain Model (Local & GCS Sync)")
    
    if st.button("🚀 Train Model with Updated Dataset"):
        log_placeholder = st.empty()
        result_placeholder = st.empty()
        log_lines = []

        def append_log(msg):
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            log_placeholder.markdown(f'<div class="log-box">{"<br>".join(log_lines[-20:])}</div>', unsafe_allow_html=True)

        try:
            import training
            importlib.reload(training)
            append_log("Starting training pipeline...")
            result = training.train_model(status_callback=append_log, sync_gcs=True, use_augmentation=True)
            
            # Upload to laptop FastAPI (always — this is the fallback source for Pi)
            append_log("Uploading new model to laptop FastAPI server...")
            import requests
            from config import API_BASE_URL, MODEL_PATH
            try:
                with open(MODEL_PATH, "rb") as f:
                    r = requests.post(f"{API_BASE_URL}/api/model/upload", files={"file": f})
                    r.raise_for_status()
                    resp = r.json()
                    append_log(f"✅ Model version {resp.get('version')} deployed to laptop FastAPI!")
            except Exception as api_err:
                append_log(f"⚠️  FastAPI upload failed: {api_err}")
                append_log("Model is saved locally. Pi must be on same network to fetch it.")
            
            # Show result with GCS status
            gcs_status = "☁️ GCS: ✅ Uploaded" if result.get('gcs_success') else "☁️ GCS: ⚠️ Failed (Pi will use laptop fallback)"
            result_placeholder.success(
                f"🎉 Training complete!\n\n"
                f"**Train acc:** {result['train_acc']:.2%} | **Test acc:** {result['test_acc']:.2%}\n\n"
                f"**Registered IDs:** {', '.join(result['classes'])}\n\n"
                f"{gcs_status}"
            )
            if not result.get('gcs_success'):
                st.warning("⚠️ GCS upload failed. The model is saved locally and on the FastAPI server. "
                           "The Pi will download it from this laptop over WiFi as a fallback.")
            
            # Force reload of local inference module for Local USB mode
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
        source = st.selectbox("📹 Video Source", [
            "Local USB — Live View", 
            "Local USB — Record & Analyze", 
            "Upload Video File",
            "Server Stream (Pi)"
        ])
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0) if "USB" in source else 0
        uploaded_video = st.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov"]) if source == "Upload Video File" else None

        st.markdown("---")
        is_running, is_recording = st.session_state["running"], st.session_state["recording"]

        if source == "Local USB — Live View":
            start_btn = st.button("▶ Start Analysis (Pi Model)", disabled=is_running)
            local_start_btn = st.button("▶ Start Local Analysis", disabled=is_running,
                                        help="Runs inference locally on this laptop. Isolated from Pi mode.")
            stop_btn = st.button("⏹ Stop Analysis", disabled=not is_running)
            record_start_btn = record_stop_btn = analyze_recorded_btn = False
            server_stream_btn = False
        elif source == "Local USB — Record & Analyze":
            record_start_btn = st.button("⏺ Start Recording", disabled=is_recording)
            record_stop_btn = st.button("⏹ Stop Recording", disabled=not is_recording)
            start_btn = stop_btn = server_stream_btn = local_start_btn = False
            analyze_recorded_btn = st.button("🔍 Analyze Last Recording", disabled=is_running) if st.session_state.get("recorded_video_path") and not is_recording else False
        elif source == "Server Stream (Pi)":
            server_stream_btn = st.button("📡 Connect to Server", disabled=is_running)
            stop_btn = st.button("⏹ Disconnect", disabled=not is_running)
            start_btn = record_start_btn = record_stop_btn = analyze_recorded_btn = local_start_btn = False
        else:
            start_btn = st.button("▶ Analyze Video", disabled=is_running)
            stop_btn = st.button("⏹ Stop", disabled=not is_running)
            record_start_btn = record_stop_btn = analyze_recorded_btn = server_stream_btn = local_start_btn = False

        st.markdown("---")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            frames_metric = st.empty()
        with mcol2:
            fps_metric = st.empty()

    with info_col:
        st.markdown("##### 🎥 Live Feeds")
        show_feed = st.toggle("📺 Show Video Feed", value=True, help="Toggle to hide/show live video (attendance still runs in background)")
        
        feed_col1, feed_col2 = st.columns(2)
        with feed_col1:
            st.markdown("**Input (Raw)**")
            raw_frame_placeholder = st.empty()
        with feed_col2:
            st.markdown("**Output (Analyzed)**")
            analyzed_frame_placeholder = st.empty()
            
        st.markdown("##### 📋 Pi Camera Attendance (from MongoDB)")
        events_placeholder = st.empty()

        st.markdown("---")
        st.markdown("##### 💻 Local Camera Attendance (from MongoDB)")
        local_events_placeholder = st.empty()
        # Populate local attendance table on page load
        local_att_logs = get_today_local_attendance()
        if local_att_logs:
            df_local = pd.DataFrame(local_att_logs)[["emp_id", "emp_name", "timestamp", "status"]]
            local_events_placeholder.dataframe(df_local.tail(10), use_container_width=True)
        else:
            local_events_placeholder.info("No local camera attendance recorded today.")

        playback_area = st.empty()

    if stop_btn or record_stop_btn:
        st.session_state["stop_flag"] = True
        st.session_state["running"] = False
        st.session_state["recording"] = False
        st.rerun()

    def show_session_videos():
        rec_path, result_path = st.session_state.get("recorded_video_path"), st.session_state.get("result_video_path")
        has_rec, has_res = rec_path and os.path.exists(rec_path), result_path and os.path.exists(result_path)
        if has_rec or has_res:
            with playback_area.container():
                st.markdown("---")
                st.markdown("### 🎬 Session Videos")
                v1, v2 = st.columns(2)
                if has_rec:
                    with v1:
                        st.markdown("**📹 Captured Video**")
                        st.video(rec_path)
                if has_res:
                    with v2:
                        st.markdown("**🔍 Analyzed Result**")
                        st.video(result_path)

    show_session_videos()

    def run_analysis_loop(cap, record_raw=False, raw_writer=None, result_writer=None, target_fps=30):
        callback_fn = import_main_callback()
        idx = 0
        frame_interval = 1.0 / target_fps
        next_write_time = time.time()
        last_ui_update = 0.0
        
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while not st.session_state["stop_flag"]:
            ret, frame = cap.read()
            if not ret: break

            if record_raw and raw_writer: raw_writer.write(frame)

            try:
                annotated = callback_fn(frame, idx)
            except Exception as e:
                annotated = frame.copy()
                cv2.putText(annotated, f"Err: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if result_writer:
                now = time.time()
                while next_write_time <= now:
                    result_writer.write(annotated)
                    next_write_time += frame_interval

            if show_feed:
                now = time.time()
                if idx % 2 == 0 and (now - last_ui_update) >= (1/12.0):
                    pr_r, pr_a = cv2.resize(frame, (480, int(480*frame.shape[0]/frame.shape[1]))), cv2.resize(annotated, (480, int(480*annotated.shape[0]/annotated.shape[1])))
                    raw_frame_placeholder.image(cv2.cvtColor(pr_r, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    analyzed_frame_placeholder.image(cv2.cvtColor(pr_a, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    last_ui_update = now
            else:
                # Clear placeholders if feed is hidden
                if idx == 0 or (idx % 30 == 0): # just call empty() periodically or once
                    raw_frame_placeholder.empty()
                    analyzed_frame_placeholder.empty()

            idx += 1
            fps_frame_count += 1
            
            # FPS calculation
            now_fps = time.time()
            if now_fps - fps_start_time >= 1.0:
                fps = fps_frame_count / (now_fps - fps_start_time)
                fps_metric.metric("⚡ Live FPS", f"{fps:.1f}")
                fps_start_time = now_fps
                fps_frame_count = 0

            frames_metric.metric("🖼️ Frames Processed", idx)

            att_logs = get_today_attendance()
            if att_logs:
                df = pd.DataFrame(att_logs)[["emp_id", "emp_name", "timestamp", "status"]]
                events_placeholder.dataframe(df.tail(10), use_container_width=True)

        return idx

    if start_btn: st.session_state["active_mode"] = "pi_live"
    elif local_start_btn: st.session_state["active_mode"] = "local_live"
    elif server_stream_btn: st.session_state["active_mode"] = "server_stream"
    elif record_start_btn: st.session_state["active_mode"] = "record_video"
    elif analyze_recorded_btn: st.session_state["active_mode"] = "analyze_video"

    if st.session_state.get("active_mode") and not st.session_state.get("stop_flag"):
        st.session_state["running"] = st.session_state["active_mode"] != "record_video"
        st.session_state["recording"] = st.session_state["active_mode"] == "record_video"
        mode = st.session_state["active_mode"]

        # ── LOCAL ANALYSIS (laptop camera, isolated from Pi) ──
        if mode == "local_live":
            import main_local as ml
            cam_idx = ml.detect_camera()
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                st.error(f"❌ Cannot open camera at index {cam_idx}.")
                st.session_state.update(running=False)
                st.stop()

            st.info(f"💻 Local Analysis running on camera index {cam_idx}")
            with st.spinner("Loading local models…"):
                callback_fn = import_local_callback()

            idx = 0
            last_ui_update = 0.0
            fps_start_time = time.time()
            fps_frame_count = 0
            
            while not st.session_state["stop_flag"]:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    annotated = callback_fn(frame, idx)
                except Exception as e:
                    annotated = frame.copy()
                    cv2.putText(annotated, f"Err: {e}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if show_feed:
                    now = time.time()
                    if idx % 2 == 0 and (now - last_ui_update) >= (1 / 12.0):
                        pr_r = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
                        pr_a = cv2.resize(annotated, (480, int(480 * annotated.shape[0] / annotated.shape[1])))
                        raw_frame_placeholder.image(cv2.cvtColor(pr_r, cv2.COLOR_BGR2RGB),
                                                    channels="RGB", use_container_width=True)
                        analyzed_frame_placeholder.image(cv2.cvtColor(pr_a, cv2.COLOR_BGR2RGB),
                                                         channels="RGB", use_container_width=True)
                        last_ui_update = now
                else:
                    if idx == 0 or (idx % 30 == 0):
                        raw_frame_placeholder.empty()
                        analyzed_frame_placeholder.empty()

                idx += 1
                fps_frame_count += 1
                
                now_fps = time.time()
                if now_fps - fps_start_time >= 1.0:
                    fps = fps_frame_count / (now_fps - fps_start_time)
                    fps_metric.metric("⚡ Live FPS", f"{fps:.1f}")
                    fps_start_time = now_fps
                    fps_frame_count = 0

                frames_metric.metric("🖼️ Frames Processed", idx)

                # Refresh local attendance table every 30 frames
                if idx % 30 == 0:
                    local_logs = get_today_local_attendance()
                    if local_logs:
                        df_local = pd.DataFrame(local_logs)[["emp_id", "emp_name", "timestamp", "status"]]
                        local_events_placeholder.dataframe(df_local.tail(10), use_container_width=True)

            cap.release()
            st.session_state.update(running=False)
            st.success(f"✅ Local Analysis stopped. {idx} frames processed.")
            st.rerun()

        elif mode == "server_stream":
            import websockets
            import asyncio
            from config import WS_BASE_URL
            
            # Setup columns for side-by-side streams
            col_raw, col_analyzed = st.columns(2)
            col_raw.subheader("Raw Feed")
            col_analyzed.subheader("Analyzed Feed")
            raw_placeholder = col_raw.empty()
            analyzed_placeholder = col_analyzed.empty()
            
            async def consume_single_stream(uri_suffix, placeholder, metric_name):
                idx = 0
                fps_start_time = time.time()
                fps_frame_count = 0
                
                uri = f"{WS_BASE_URL}/ws/stream/ui/{uri_suffix}"
                try:
                    async with websockets.connect(uri) as ws:
                        while not st.session_state["stop_flag"]:
                            try:
                                data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                # L1 FIX: Drain all stale frames, keep only the newest
                                while True:
                                    try:
                                        data = await asyncio.wait_for(ws.recv(), timeout=0.005)
                                    except asyncio.TimeoutError:
                                        break
                                
                                if show_feed:
                                    nparr = np.frombuffer(data, np.uint8)
                                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                                else:
                                    if idx == 0 or (idx % 30 == 0):
                                        placeholder.empty()
                                
                                idx += 1
                                if uri_suffix == "analyzed":
                                    fps_frame_count += 1
                                    now_fps = time.time()
                                    if now_fps - fps_start_time >= 1.0:
                                        fps = fps_frame_count / (now_fps - fps_start_time)
                                        fps_metric.metric("⚡ Live FPS", f"{fps:.1f}")
                                        fps_start_time = now_fps
                                        fps_frame_count = 0
                                    
                                    frames_metric.metric("🖼️ Frames Processed", idx)
                                    
                                    # Refresh attendance board
                                    if idx % 30 == 0:
                                        att_logs = get_today_attendance()
                                        if att_logs:
                                            df = pd.DataFrame(att_logs)[["emp_id", "emp_name", "timestamp", "status"]]
                                            events_placeholder.dataframe(df.tail(10), use_container_width=True)
                            except asyncio.TimeoutError:
                                continue
                except Exception as e:
                    placeholder.error(f"Stream '{uri_suffix}' disconnected: {e}")

            async def consume_both_streams():
                st.success("Connecting to dual Server Streams...")
                task1 = asyncio.create_task(consume_single_stream("raw", raw_placeholder, "Raw Frames"))
                task2 = asyncio.create_task(consume_single_stream("analyzed", analyzed_placeholder, "Analyzed Frames"))
                await asyncio.gather(task1, task2)

            asyncio.run(consume_both_streams())
            st.rerun()

        elif mode == "analyze_video":
            cap = cv2.VideoCapture(st.session_state["recorded_video_path"])
        elif source == "Upload Video File" and mode == "pi_live": # wait, pi_live is used as default when Analyze Video is clicked
            # Let's check: button was start_btn, so mode is pi_live.
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = os.path.join(RECORDINGS_DIR, f"uploaded_{ts}.mp4")
            with open(raw_path, "wb") as f: f.write(uploaded_video.read())
            st.session_state["recorded_video_path"] = raw_path
            cap = cv2.VideoCapture(raw_path)
        else: # pi_live on normal camera or record_video
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if mode not in ["server_stream", "local_live"]:
            if not cap.isOpened():
                st.error("❌ Cannot open video source.")
                st.session_state.update(running=False, recording=False)
                st.session_state["active_mode"] = None
                st.stop()

            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            res_path = os.path.join(RECORDINGS_DIR, f"result_{ts}.mp4")
            res_writer = cv2.VideoWriter(res_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            
            raw_writer = None
            if mode == "record_video":
                raw_path = os.path.join(RECORDINGS_DIR, f"recorded_{ts}.mp4")
                raw_writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                st.session_state["recorded_video_path"] = raw_path

            with st.spinner("Loading models…"): import_main_callback()
            
            n = run_analysis_loop(cap, record_raw=(mode=="record_video"), raw_writer=raw_writer, result_writer=res_writer)
            
            cap.release()
            res_writer.release()
            if raw_writer: raw_writer.release()
            
            st.session_state.update(running=False, recording=False, result_video_path=res_path, active_mode=None)
            st.success(f"✅ Processing complete. {n} frames.")
            st.rerun()
