import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; font-size: clamp(1.5rem, 5vw, 2.5rem) !important; word-break: keep-all !important; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; color: white; }
    .form-card { background-color: #1E252D; padding: 15px; border-radius: 10px; border-left: 5px solid #00E5FF; margin-top: 10px; color: white; }
    .est-card { background-color: #2D241E; padding: 20px; border-radius: 10px; border-left: 5px solid #FFC107; margin-top: 15px; color: white; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ App Settings")
    tracking_mode = st.radio("Mode:", ["⚡ Quick Track", "📈 1RM Profiler"])
    
    st.markdown("---")
    st.subheader("👤 Lifter Profile")
    profile = st.select_slider(
        "Select your 'Feel':",
        options=["Grinder", "Standard", "Explosive"],
        value="Standard",
        help="Explosive: High speed on warmups, sharp drop-off. Grinder: Constant speed, slow but strong."
    )
    
    # --- SWAPPED VALUES FOR CORRECT LOGIC ---
    # Explosive (High Sensitivity) = Higher drop in %1RM per m/s = Lower 1RM Estimate
    # Grinder (Low Sensitivity) = Lower drop in %1RM per m/s = Higher 1RM Estimate
    sensitivity_map = {"Grinder": 0.35, "Standard": 0.55, "Explosive": 0.95}
    SENSITIVITY = sensitivity_map[profile]

# --- SESSION STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'workout_log' not in st.session_state: st.session_state.workout_log = []

# --- UPLOAD PHASE ---
if not st.session_state.tracking_done:
    col_w, col_u = st.columns([1, 2])
    with col_w:
        label = "Weight (Optional)" if "Quick" in tracking_mode else "Weight (Required)"
        weight_lifted = st.number_input(label, min_value=0.0, value=0.0, step=5.0)
    with col_u:
        uploaded_file = st.file_uploader("Upload MP4 or MOV", type=["mp4", "mov"])

    if uploaded_file and (weight_lifted > 0 or "Quick" in tracking_mode):
        tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
        with open(tpath, "wb") as f: f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tpath); fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if ret:
            orig_h, orig_w = first_frame.shape[:2]; display_w = 400; scale_factor = orig_w / display_w
            frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            first_frame_res = cv2.resize(frame_rgb, (display_w, int(display_w * (orig_h / orig_w))))
            if not st.session_state.clicked:
                st.markdown("### 🎯 Step 1: Click the Barbell Edge")
                value = streamlit_image_coordinates(first_frame_res, key="clicker")
                if value:
                    gray = cv2.cvtColor(first_frame_res, cv2.COLOR_RGB2GRAY)
                    y1, y2, x1, x2 = max(0, value['y']-10), min(first_frame_res.shape[0], value['y']+10), max(0, value['x']-10), min(display_w, value['x']+10)
                    roi = gray[y1:y2, x1:x2]
                    mag = cv2.magnitude(cv2.Sobel(roi, cv2.CV_64F, 1, 0), cv2.Sobel(roi, cv2.CV_64F, 0, 1))
                    _, _, _, max_loc = cv2.minMaxLoc(mag)
                    st.session_state.coords = (x1 + max_loc[0], y1 + max_loc[1]); st.session_state.clicked = True; st.rerun()

            if st.session_state.clicked:
                cx, cy = st.session_state.coords; tracker = cv2.TrackerCSRT_create()
                orig_cx, orig_cy = int(cx * scale_factor), int(cy * scale_factor); box_size = int(50 * scale_factor)
                tracker.init(first_frame, (orig_cx - box_size//2, orig_cy - box_size//2, box_size, box_size))
                x_hist_orig, y_hist_orig, bboxes_orig, frames_display = [], [], [], []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); progress = st.progress(0)
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    ok, box = tracker.update(frame)
                    bx, by, bw, bh = [int(v) for v in (box if ok else bboxes_orig[-1])]
                    x_hist_orig.append(bx + bw//2); y_hist_orig.append(by + bh//2); bboxes_orig.append((bx, by, bw, bh))
                    frames_display.append(cv2.cvtColor(cv2.resize(frame, (display_w, int(display_w * (orig_h / orig_w)))), cv2.COLOR_BGR2RGB))
                    progress.progress((i + 1) / total_frames)

                m_per_px = 0.45 / bboxes_orig[0][3]
                v_instant = [(y_hist_orig[j-1] - y_hist_orig[j]) * m_per_px * fps if j > 0 else 0 for j in range(len(y_hist_orig))]
                v_smooth = [np.mean(v_instant[max(0, x-3):min(len(v_instant), x+3)]) for x in range(len(v_instant))]
                rep_data, is_moving, start_f = [], False, 0
                for i, v in enumerate(v_smooth):
                    if not is_moving and v > 0.15: is_moving, start_f = True, i
                    elif is_moving and v <= 0:
                        end_f = i
                        if (y_hist_orig[start_f] - y_hist_orig[end_f]) * m_per_px > 0.30:
                            x_coords = x_hist_orig[start_f:end_f+1]
                            drift_m = (max(x_coords) - min(x_coords)) * m_per_px
                            rep_data.append({"id": len(rep_data)+1, "start": start_f, "end": end_f, "avg_v": np.mean(v_smooth[start_f:end_f+1]), "dur": (end_f - start_f)/fps, "drift": drift_m})
                        is_moving = False
                if rep_data and "Profiler" in tracking_mode: st.session_state.workout_log.append({"weight": weight_lifted, "velocity": max([r['avg_v'] for r in rep_data])})

                path_pts_disp, out_frames = [], []
                for i, f in enumerate(frames_display):
                    bx, by, bw, bh = [int(v / scale_factor) for v in bboxes_orig[i]]
                    path_pts_disp.append((bx+bw//2, by+bh//2))
                    active = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                    if len(path_pts_disp) > 1:
                        for j in range(max(1, i-60), len(path_pts_disp)):
                            color = (255, 75, 173) if path_pts_disp[j][1] < path_pts_disp[j-1][1] else (0, 255, 255)
                            cv2.line(f, path_pts_disp[j-1], path_pts_disp[j], color, 2)
                    if active:
                        cv2.line(f, (path_pts_disp[active['start']][0], 0), (path_pts_disp[active['start']][0], f.shape[0]), (255,255,255), 1, cv2.LINE_AA)
                        cv2.rectangle(f, (0,0), (display_w, 50), (0,0,0), -1)
                        cv2.putText(f, f"REP {active['id']} | DRIFT: {active['drift']*39.37:.1f}in", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,229,255), 2)
                    out_frames.append(f)

                final_p = os.path.join(tempfile.gettempdir(), "out.mp4"); imageio.mimsave(final_p, out_frames, fps=fps, codec='libx264')
                st.session_state.rep_data, st.session_state.final_vid_path, st.session_state.last_weight, st.session_state.tracking_done = rep_data, final_p, weight_lifted, True; st.rerun()

# --- DASHBOARD PHASE ---
if st.session_state.tracking_done:
    c1, c2 = st.columns([2, 1.5])
    with c1:
        st.video(st.session_state.final_vid_path)
        if st.button("➕ Track Next Set"): st.session_state.clicked = False; st.session_state.tracking_done = False; st.rerun()
    with c2:
        st.subheader("📊 Performance Data")
        for r in st.session_state.rep_data:
            st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} m/s | {r["dur"]:.2f}s</div>', unsafe_allow_html=True)
            drift_in = r['drift'] * 39.37
            grade = "ELITE" if drift_in < 2 else "STABLE" if drift_in < 4 else "LEAKAGE"
            st.markdown(f'<div class="form-card">⚖️ <b>FORM GRADE: {grade}</b><br>Horizontal Drift: {drift_in:.1f} inches</div>', unsafe_allow_html=True)
        
        if st.session_state.last_weight > 0 and "Quick" in tracking_mode:
            best_v = max([r['avg_v'] for r in st.session_state.rep_data])
            # CORRECTED MATH LOGIC
            velocity_reserve = max(0, best_v - 0.30)
            # Higher SENSITIVITY now properly results in lower est_pct, making 1RM lower (conservative)
            est_pct = 1.0 - (velocity_reserve * SENSITIVITY)
            est_pct = max(0.35, est_pct)
            est_1rm = st.session_state.last_weight / est_pct
            st.markdown(f'<div class="est-card">🟡 <b>{profile.upper()} 1RM EST.</b><br><span style="font-size: 2.2em; color: #FFC107;">{est_1rm:.1f}</span></div>', unsafe_allow_html=True)
