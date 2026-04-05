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

# --- UI REFINEMENT: THE "CLEAN LOOK" ---
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { font-weight: 900; text-align: center; text-transform: uppercase; letter-spacing: -1px; margin-bottom: 0px;}
    .video-container { max-width: 500px; margin: 0 auto; border-radius: 15px; overflow: hidden; border: 1px solid #333; }
    
    .card { background-color: #1A1A1A; border: 1px solid #333; padding: 20px; border-radius: 12px; margin-bottom: 10px; }
    .stat-label { color: #888; font-size: 0.8em; text-transform: uppercase; font-weight: 700; display: block; margin-bottom: 5px; }
    .stat-value { font-size: 2.2em; font-weight: 900; color: #FFFFFF; }
    
    .rep-strip { border-left: 4px solid #FF1E56; padding-left: 15px; margin-bottom: 15px; }
    .form-strip { border-left: 4px solid #00D2FF; padding-left: 15px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; margin-bottom: 25px;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

# --- STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

EXERCISES = ["Deadlift", "Squat", "Bench Press", "Overhead Press"]
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
                            x_hist.append(box[0]+box[2]/2); y_hist.append(box[1]+box[3]/2); bboxes.append(box)
                            # Draw path & center line
                            cv2.line(frame, (int(st.session_state.coords[0]), 0), (int(st.session_state.coords[0]), h), (100, 100, 100), 1)
                            if len(x_hist) > 1:
                                for j in range(1, len(x_hist)):
                                    cv2.line(frame, (int(x_hist[j-1]), int(y_hist[j-1])), (int(x_hist[j]), int(y_hist[j])), (0, 255, 0), 2)
                        
                        frames_out.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                        progress.progress((i + 1) / total_frames)
                    
                    # --- NOISE-FILTERING REP LOGIC ---
                    m_per_px = 0.45 / bboxes[0][3]
                    v_instant = [(y_hist[j-1]-y_hist[j])*m_per_px*fps if j>0 else 0 for j in range(len(y_hist))]
                    
                    reps_found = []
                    is_moving = False; start_idx = 0
                    for i, v in enumerate(v_instant):
                        # Filter: Must be moving faster than 0.15 m/s to count as a rep
                        if not is_moving and v > 0.15: 
                            is_moving, start_idx = True, i
                        elif is_moving and v < 0:
                            duration = (i - start_idx) / fps
                            # Filter: Rep must last at least 0.4 seconds (removes bounces/drops)
                            if duration > 0.4:
                                rep_v = np.mean(v_instant[start_idx:i])
                                reps_found.append({"v": round(rep_v, 2), "dur": round(duration, 2)})
                            is_moving = False

                    drift_in = round((max(x_hist) - min(x_hist)) * m_per_px * 39.37, 1)
                    avg_v = reps_found[0]["v"] if reps_found else 0
                    est_rpe = round(max(6.0, min(10.0, 10.5 - (avg_v * 4))) * 2) / 2
                    
                    out_path = os.path.join(tempfile.gettempdir(), "tracked.mp4")
                    imageio.mimsave(out_path, frames_out, fps=fps, codec='libx264')
                    
                    st.session_state.rep_data = {"reps": reps_found, "ai_rpe": est_rpe, "video": out_path, "drift": drift_in}
                    st.session_state.tracking_done = True; st.rerun()

    if st.session_state.tracking_done:
        res = st.session_state.rep_data
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(res["video"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- CLEAN PERFORMANCE DATA VIEW ---
        st.markdown("### 📊 Performance Data")
        for idx, r in enumerate(res["reps"]):
            st.markdown(f'<div class="card rep-strip"><span class="stat-label">REP {idx+1}</span><span style="font-size:1.2em; font-weight:700;">{r["v"]} m/s | {r["dur"]}s</span></div>', unsafe_allow_html=True)
        
        grade = "STABLE" if res["drift"] < 4 else "LEAKAGE"
        st.markdown(f'<div class="card form-strip"><span class="stat-label">FORM GRADE: {grade}</span><span style="font-size:1.2em; font-weight:700;">Drift: {res["drift"]} inches</span></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card" style="border-top: 4px solid #FF1E56; text-align: center;"><span class="stat-label">AI RPE</span><div class="stat-value">{res["ai_rpe"]}</div><span style="color:#555; font-size:0.7em;">Single-Rep Proximity</span></div>', unsafe_allow_html=True)
        
        # Adjustable Section
        user_rpe = st.slider("Manual Override", 5.0, 10.0, float(res["ai_rpe"]), 0.5)
        rir = 10.0 - user_rpe
        adj_1rm = round(weight * (36 / (37 - (1 + rir))), 1)
        
        st.markdown(f'<div class="card" style="border-top: 4px solid #00D2FF; text-align: center;"><span class="stat-label">AI 1RM EST</span><div class="stat-value" style="color:#00D2FF;">{adj_1rm}</div><span style="color:#555; font-size:0.7em;">Brzycki Hybrid</span></div>', unsafe_allow_html=True)

        if st.button("💾 SAVE TO VAULT", use_container_width=True):
            supabase.table("lifts").insert({"exercise": exercise, "weight": weight, "reps": len(res["reps"]), "rpe": user_rpe, "est_1rm": adj_1rm}).execute()
            st.success("Set Archived."); st.session_state.tracking_done = False; st.session_state.clicked = False; st.session_state.uploader_key += 1; st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    logs = supabase.table("lifts").select("*").order("created_at", desc=True).execute()
    if logs.data:
        for row in logs.data:
            st.markdown(f"**{row['exercise']}**: {row['weight']} lbs x {row['reps']} | RPE {row['rpe']} | 1RM {row['est_1rm']}")
            st.markdown("---")

st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #333; font-size: 0.7em; font-weight: 800;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
