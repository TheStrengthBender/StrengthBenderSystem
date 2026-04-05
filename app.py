import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
from datetime import datetime
from supabase import create_client, Client
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CLOUD CONNECTION ---
SUPABASE_URL = "https://ohnzahpomkezpogfhcvj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obnphaHBvbWtlenBvZ2ZoY3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU0MjAyMjcsImV4cCI6MjA5MDk5NjIyN30.d0kvd-bEKWeVsnmXf9UvWyQsCUuSYOubNgnlUpt-vqw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="IRON SIGHT", page_icon="🎯", layout="wide")

# --- CSS (UNCHANGED) ---
st.markdown("""<style>
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF !important; font-weight: 900; text-align: center; text-transform: uppercase; margin-bottom: 0px;}
    .rep-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #E63946; margin-bottom: 10px; color: white; }
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    </style>""", unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)

# --- STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

EXERCISES = ["Squat", "Bench Press", "Deadlift", "Overhead Press", "Barbell Row"]
tab1, tab2 = st.tabs(["⚡ LIVE TRACK", "🗄️ TACTICAL ARCHIVE"])

with tab1:
    col_ex, col_w = st.columns(2)
    with col_ex: exercise = st.selectbox("Exercise", EXERCISES)
    with col_w: weight = st.number_input("Weight (lbs)", min_value=0.0, step=5.0)

    if not st.session_state.tracking_done:
        uploaded_file = st.file_uploader("Upload Set", type=["mp4", "mov"], key=f"uploader_{st.session_state.uploader_key}")

        if uploaded_file and weight > 0:
            tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
            with open(tpath, "wb") as f: f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            
            if ret:
                h, w = first_frame.shape[:2]
                display_w = 480
                display_h = int(display_w * (h / w))
                first_frame_res = cv2.resize(first_frame, (display_w, display_h))
                
                if not st.session_state.clicked:
                    st.markdown("### 🎯 Lock the Target")
                    value = streamlit_image_coordinates(cv2.cvtColor(first_frame_res, cv2.COLOR_BGR2RGB), key="clicker")
                    if value:
                        st.session_state.coords = (value['x'] * (w/display_w), value['y'] * (h/display_h))
                        st.session_state.clicked = True; st.rerun()

                if st.session_state.clicked:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(first_frame, (int(st.session_state.coords[0]-25), int(st.session_state.coords[1]-25), 50, 50))
                    
                    y_hist, x_hist, bboxes, frames_out = [], [], [], []
                    progress = st.progress(0)
                    
                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret: break
                        ok, box = tracker.update(frame)
                        if ok:
                            bx, by, bw, bh = [int(v) for v in box]
                            cx, cy = bx + bw//2, by + bh//2
                            x_hist.append(cx); y_hist.append(cy); bboxes.append(box)
                            
                            # Draw Bar Path
                            if len(x_hist) > 1:
                                for j in range(1, len(x_hist)):
                                    color = (0, 255, 0) if y_hist[j] < y_hist[j-1] else (255, 255, 255)
                                    cv2.line(frame, (x_hist[j-1], y_hist[j-1]), (x_hist[j], y_hist[j]), color, 3)
                        
                        frames_out.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                        progress.progress((i + 1) / total_frames)
                    
                    # Math & Rep Detection
                    m_per_px = 0.45 / bboxes[0][3]
                    v_smooth = [np.mean([(y_hist[j-1]-y_hist[j])*m_per_px*fps for j in range(max(1, i-2), i+1)]) for i in range(len(y_hist))]
                    
                    # Simple Rep Logic
                    avg_v = np.mean([v for v in v_smooth if v > 0.1]) if any(v > 0.1 for v in v_smooth) else 0
                    est_rpe = max(6.0, min(10.0, 10.5 - (avg_v * 4)))
                    est_1rm = weight * (36 / (37 - (1 + (10-est_rpe))))

                    # Save analyzed video to temp file
                    out_path = os.path.join(tempfile.gettempdir(), "tracked.mp4")
                    imageio.mimsave(out_path, frames_out, fps=fps/2, codec='libx264')
                    
                    st.session_state.rep_data = {"v": avg_v, "rpe": est_rpe, "1rm": est_1rm, "video": out_path}
                    st.session_state.tracking_done = True; st.rerun()

    if st.session_state.tracking_done:
        res = st.session_state.rep_data
        st.video(res["video"])
        
        st.markdown(f'<div class="rep-card"><b>AVG VELOCITY:</b> {res["v"]:.2f} m/s</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="stat-card-red">AI RPE<br><h2>{res["rpe"]}</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-card-green">EST 1RM<br><h2>{res["1rm"]:.1f}</h2></div>', unsafe_allow_html=True)

        if st.button("💾 SAVE TO CLOUD ARCHIVE", use_container_width=True):
            entry = {"exercise": exercise, "weight": weight, "reps": 1, "rpe": res["rpe"], "est_1rm": res["1rm"]}
            supabase.table("lifts").insert(entry).execute()
            st.success("Set Locked."); st.session_state.tracking_done = False; st.session_state.clicked = False; st.session_state.uploader_key += 1; st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    logs = supabase.table("lifts").select("*").order("created_at", desc=True).execute()
    if logs.data:
        for row in logs.data:
            with st.container():
                c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
                with c1: st.write(f"🏋️ **{row['exercise']}**")
                with c2: st.write(f"{row['weight']} lbs")
                with c3: st.write(f"RPE {row['rpe']}")
                with c4: st.write(f"🎯 **{row['est_1rm']:.1f}**")
                st.markdown("---")

st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
