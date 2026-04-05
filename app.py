import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="IRON SIGHT", page_icon="🎯", layout="wide")

# --- CUSTOM CSS: THE IRON SIGHT AESTHETIC ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', 'Roboto', sans-serif; }
    
    /* Header Typography */
    h1 { color: #FFFFFF !important; font-weight: 900; font-size: clamp(2rem, 6vw, 3rem) !important; text-align: center; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0px;}
    
    /* Tactical Cards */
    .rep-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #E63946; margin-bottom: 10px; color: white; }
    .form-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #8B949E; margin-bottom: 10px; color: white; }
    
    /* Stats Row (Red & Green Top Strokes) */
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    
    /* Adjusted 1RM Card */
    .est-card-gold { background-color: #2D3139; padding: 20px; border-radius: 8px; border-left: 5px solid #FFC107; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    
    /* Warning Text */
    .warning-text { color: #E63946; font-weight: bold; font-size: 0.9em; text-align: center; margin-top: -10px; margin-bottom: 15px; }
    
    /* Sidebar styling to match */
    [data-testid="stSidebar"] { background-color: #121316; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E; margin-bottom: 30px;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'workout_log' not in st.session_state: st.session_state.workout_log = []
if 'last_weight' not in st.session_state: st.session_state.last_weight = 0.0
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

with st.sidebar:
    st.header("⚙️ App Settings")
    tracking_mode = st.radio("Mode:", ["⚡ Quick Track", "📈 1RM Profiler"])
    st.markdown("---")
    st.info("💡 **Pro-Tip:** Shoot from a direct side angle and ensure the camera is stable for the most accurate physics tracking.")

# --- UPLOAD PHASE ---
if not st.session_state.tracking_done:
    col_w, col_u = st.columns([1, 2])
    with col_w:
        weight_in = st.number_input("Weight on Bar (lbs)", min_value=0.0, value=st.session_state.last_weight, step=5.0)
        st.session_state.last_weight = weight_in
    with col_u:
        uploaded_file = st.file_uploader("Upload MP4 or MOV", type=["mp4", "mov"], key=f"uploader_{st.session_state.uploader_key}")
        
        if not uploaded_file:
            st.markdown("### 👋 Welcome to the Lab")
            st.markdown("1. Enter the weight on the bar.\n2. Upload a video of your set.\n3. Click the plate to extract your data.")

    if uploaded_file and (st.session_state.last_weight > 0 or "Quick" in tracking_mode):
        tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
        with open(tpath, "wb") as f: f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        
        if ret:
            orig_h, orig_w = first_frame.shape[:2]
            
            # Turbo processing dimensions
            track_w = 640 
            track_h = int(track_w * (orig_h / orig_w))
            first_frame_track = cv2.resize(first_frame, (track_w, track_h))
            
            display_w = 320 
            display_h = int(display_w * (orig_h / orig_w))
            frame_rgb = cv2.cvtColor(first_frame_track, cv2.COLOR_BGR2RGB)
            first_frame_res = cv2.resize(frame_rgb, (display_w, display_h))
            
            click_to_track_ratio = track_w / display_w
            
            if not st.session_state.clicked:
                st.markdown("### 🎯 Lock the Target")
                st.caption("Click the center of the barbell plate to begin AI tracking.")
                value = streamlit_image_coordinates(first_frame_res, key="clicker")
                if value:
                    gray = cv2.cvtColor(first_frame_res, cv2.COLOR_RGB2GRAY)
                    y1, y2, x1, x2 = max(0, value['y']-10), min(first_frame_res.shape[0], value['y']+10), max(0, value['x']-10), min(display_w, value['x']+10)
                    roi = gray[y1:y2, x1:x2]
                    mag = cv2.magnitude(cv2.Sobel(roi, cv2.CV_64F, 1, 0), cv2.Sobel(roi, cv2.CV_64F, 0, 1))
                    _, _, _, max_loc = cv2.minMaxLoc(mag)
                    st.session_state.coords = (x1 + max_loc[0], y1 + max_loc[1]); st.session_state.clicked = True; st.rerun()

            if st.session_state.clicked:
                cx, cy = st.session_state.coords
                orig_cx = int(cx * click_to_track_ratio)
                orig_cy = int(cy * click_to_track_ratio)
                
                tracker = cv2.TrackerCSRT_create()
                box_size = int(40 * click_to_track_ratio) 
                tracker.init(first_frame_track, (orig_cx - box_size//2, orig_cy - box_size//2, box_size, box_size))
                
                x_hist_orig, y_hist_orig, bboxes_orig, frames_display = [], [], [], []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                status_box = st.info("⚡ Tracking Engaged. Analyzing Biomechanics...")
                st.markdown('<div class="warning-text">⚠️ Do not close or minimize the app while processing!</div>', unsafe_allow_html=True)
                progress = st.progress(0)
                
                for i in range(total_frames):
                    ret, raw_frame = cap.read()
                    if not ret: break
                    
                    frame = cv2.resize(raw_frame, (track_w, track_h))
                    ok, box = tracker.update(frame)
                    
                    if ok:
                        bx, by, bw, bh = [int(v) for v in box]
                    else:
                        bx, by, bw, bh = bboxes_orig[-1] if bboxes_orig else (orig_cx - box_size//2, orig_cy - box_size//2, box_size, box_size)
                    
                    x_hist_orig.append(bx + bw//2)
                    y_hist_orig.append(by + bh//2)
                    bboxes_orig.append((bx, by, bw, bh))
                    
                    if i % 2 == 0 or i == total_frames - 1:
                        frames_display.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                    
                    progress.progress((i + 1) / total_frames)
                
                status_box.empty()

                m_per_px = 0.45 / bboxes_orig[0][3]
                
                v_instant = [(y_hist_orig[j-1] - y_hist_orig[j]) * m_per_px * fps if j > 0 else 0 for j in range(len(y_hist_orig))]
                v_smooth = [np.mean(v_instant[max(0, x-3):min(len(v_instant), x+3)]) for x in range(len(v_instant))]
                
                rep_data, is_moving, start_f = [], False, 0
                for i, v in enumerate(v_smooth):
                    if not is_moving and v > 0.15: is_moving, start_f = True, i
                    elif is_moving and v <= 0:
                        end_f = i
                        dy = (y_hist_orig[start_f] - y_hist_orig[end_f]) * m_per_px 
                        dx = abs(x_hist_orig[start_f] - x_hist_orig[end_f]) * m_per_px 
                        
                        if dy > 0.15 and dx < dy:
                            x_coords = x_hist_orig[start_f:end_f+1]
                            drift_m = (max(x_coords) - min(x_coords)) * m_per_px
                            rep_data.append({"id": len(rep_data)+1, "start": start_f, "end": end_f, "avg_v": np.mean(v_smooth[start_f:end_f+1]), "dur": (end_f - start_f)/fps, "drift": drift_m})
                        
                        is_moving = False
                
                if rep_data and "Profiler" in tracking_mode: st.session_state.workout_log.append({"weight": st.session_state.last_weight, "velocity": max([r['avg_v'] for r in rep_data])})

                path_pts_disp, out_frames = [], []
                display_fps = fps / 2 
                
                for i, f in enumerate(frames_display):
                    orig_idx = i * 2 if i * 2 < len(bboxes_orig) else len(bboxes_orig) - 1
                    bx, by, bw, bh = [int(v * (display_w / track_w)) for v in bboxes_orig[orig_idx]]
                    path_pts_disp.append((bx+bw//2, by+bh//2))
                    
                    active = next((r for r in rep_data if r['start'] <= orig_idx <= r['end']), None)
                    if len(path_pts_disp) > 1:
                        for j in range(max(1, i-30), len(path_pts_disp)):
                            # Terminal Green tracking lines
                            color = (0, 255, 0) if path_pts_disp[j][1] < path_pts_disp[j-1][1] else (255, 255, 255)
                            cv2.line(f, path_pts_disp[j-1], path_pts_disp[j], color, 2)
                    if active:
                        cv2.line(f, (path_pts_disp[active['start']//2][0], 0), (path_pts_disp[active['start']//2][0], f.shape[0]), (255,255,255), 1, cv2.LINE_AA)
                        cv2.rectangle(f, (0,0), (display_w, 50), (0,0,0), -1)
                        cv2.putText(f, f"REP {active['id']} | {(orig_idx-active['start'])/fps:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    out_frames.append(f)

                final_p = os.path.join(tempfile.gettempdir(), "out.mp4"); imageio.mimsave(final_p, out_frames, fps=display_fps, codec='libx264')
                st.session_state.rep_data, st.session_state.final_vid_path, st.session_state.tracking_done = rep_data, final_p, True; st.rerun()

# --- DASHBOARD PHASE ---
if st.session_state.tracking_done:
    c1, c2 = st.columns([2, 1.5])
    with c1:
        st.video(st.session_state.final_vid_path)
        
        if st.button("🔄 TRACK NEW SET", use_container_width=True): 
            st.session_state.clicked = False
            st.session_state.tracking_done = False
            st.session_state.uploader_key += 1 
            st.rerun()
            
    with c2:
        st.subheader("📊 Performance Data")
        
        if not st.session_state.rep_data:
            st.warning("⚠️ No completed reps detected. Ensure the bar moved upwards at least 6 inches, or try a different video.")
        else:
            for r in st.session_state.rep_data:
                st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} m/s | {r["dur"]:.2f}s</div>', unsafe_allow_html=True)
                drift_in = r['drift'] * 39.37
                grade = "ELITE" if drift_in < 2 else "STABLE" if drift_in < 4 else "LEAKAGE"
                st.markdown(f'<div class="form-card">⚖️ <b>FORM GRADE: {grade}</b><br>Drift: {drift_in:.1f} inches</div>', unsafe_allow_html=True)
            
            # --- AI RPE ESTIMATION ---
            v_list = [r['avg_v'] for r in st.session_state.rep_data]
            v_max = max(v_list)
            v_last = v_list[-1]
            
            if len(st.session_state.rep_data) > 1:
                v_loss = (1 - (v_last / v_max)) * 100
                multiplier = 0.10 
                est_rpe = 5.5 + (v_loss * multiplier)
                sub_text = f"Velocity Loss: {v_loss:.1f}%"
            else:
                if v_max >= 0.80: est_rpe = 6.0
                elif v_max >= 0.65: est_rpe = 7.0
                elif v_max >= 0.50: est_rpe = 8.0
                elif v_max >= 0.40: est_rpe = 9.0
                else: est_rpe = 10.0
                sub_text = "Single-Rep Proximity"

            final_rpe = min(10.0, round(est_rpe * 2) / 2)
            safe_rpe = max(5.0, min(10.0, final_rpe))

            # --- AI BRZYCKI CALCULATION ---
            reps_performed = len(st.session_state.rep_data)
            ai_rir = 10.0 - final_rpe
            ai_effective_reps = reps_performed + ai_rir
            
            if ai_effective_reps <= 1.0:
                ai_1rm = st.session_state.last_weight
            else:
                ai_1rm = st.session_state.last_weight * (36.0 / (37.0 - ai_effective_reps))

            # UI: Side-by-Side Iron Sight Cards
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f'<div class="stat-card-red"><span style="color: #8B949E; font-size: 0.8em; font-weight: bold;">AI RPE</span><br><span style="font-size: 2.2em; font-weight: 900; color: white;">{final_rpe}</span><br><span style="color: #E63946; font-size: 0.75em;">{sub_text}</span></div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(f'<div class="stat-card-green"><span style="color: #8B949E; font-size: 0.8em; font-weight: bold;">EST 1RM</span><br><span style="font-size: 2.2em; font-weight: 900; color: #00FF00;">{ai_1rm:.1f}</span><br><span style="color: #8B949E; font-size: 0.75em;">Brzycki Hybrid</span></div>', unsafe_allow_html=True)

            # --- MANUAL OVERRIDE ---
            if st.session_state.last_weight > 0 and "Quick" in tracking_mode:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🎛️ Adjust Perceived RPE")
                
                user_rpe = st.slider("", min_value=5.0, max_value=10.0, value=float(safe_rpe), step=0.5, label_visibility="collapsed")
                
                if user_rpe != final_rpe:
                    user_rir = 10.0 - user_rpe
                    user_effective_reps = reps_performed + user_rir
                    
                    if user_effective_reps <= 1.0:
                        user_1rm = st.session_state.last_weight
                    else:
                        user_1rm = st.session_state.last_weight * (36.0 / (37.0 - user_effective_reps))
                    
                    st.markdown(f'<div class="est-card-gold"><span style="color: #8B949E; font-size: 0.9em; font-weight: bold;">ADJUSTED 1RM</span><br><span style="font-size: 2.6em; color: #FFC107; font-weight: 900;">{user_1rm:.1f} lbs</span></div>', unsafe_allow_html=True)
