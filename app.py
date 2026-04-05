import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CLOUD CONNECTION ---
SUPABASE_URL = "https://ohnzahpomkezpogfhcvj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obnphaHBvbWtlenBvZ2ZoY3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU0MjAyMjcsImV4cCI6MjA5MDk5NjIyN30.d0kvd-bEKWeVsnmXf9UvWyQsCUuSYOubNgnlUpt-vqw"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="IRON SIGHT", page_icon="🎯", layout="wide")

# --- CSS: FIXED SCALING & THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF !important; font-weight: 900; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px;}
    
    /* Video Scaling Fix */
    .video-container { max-width: 500px; margin: 0 auto; border-radius: 12px; overflow: hidden; border: 2px solid #2D3139; }
    
    /* Tactical UI Elements */
    .rep-card { background-color: #2D3139; padding: 12px; border-radius: 8px; border-left: 4px solid #E63946; margin-bottom: 8px; font-size: 0.9em; }
    .form-card { background-color: #2D3139; padding: 12px; border-radius: 8px; border-left: 4px solid #8B949E; margin-bottom: 8px; }
    .stat-card-red { background-color: #2D3139; padding: 15px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 15px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    .adj-card-gold { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 5px solid #FFC107; text-align: center; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E; margin-bottom: 20px; letter-spacing: 2px; font-size: 0.8em;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

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
        uploaded_file = st.file_uploader("Upload Set", type=["mp4", "mov"], key=f_uploader_{st.session_state.uploader_key})

        if uploaded_file and weight > 0:
            tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
            with open(tpath, "wb") as f: f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tpath)
            fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            
            if ret:
                h, w = first_frame.shape[:2]
                display_w = 480; display_h = int(display_w * (h / w))
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
                            
                            # Draw Grid & Path
                            cv2.line(frame, (int(st.session_state.coords[0]), 0), (int(st.session_state.coords[0]), h), (255, 255, 255), 1)
                            if len(x_hist) > 1:
                                for j in range(1, len(x_hist)):
                                    color = (0, 255, 0) if y_hist[j] < y_hist[j-1] else (255, 255, 255)
                                    cv2.line(frame, (x_hist[j-1], y_hist[j-1]), (x_hist[j], y_hist[j]), color, 3)
                        
                        frames_out.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                        progress.progress((i + 1) / total_frames)
                    
                    # --- REP BY REP ANALYTICS ---
                    m_per_px = 0.45 / bboxes[0][3]
                    v_instant = [(y_hist[j-1]-y_hist[j])*m_per_px*fps if j>0 else 0 for j in range(len(y_hist))]
                    
                    # Detect upward movements
                    reps_found = []
                    is_moving = False; start_idx = 0
                    for i, v in enumerate(v_instant):
                        if not is_moving and v > 0.1: is_moving, start_idx = True, i
                        elif is_moving and v < 0:
                            duration = (i - start_idx) / fps
                            if duration > 0.3:
                                reps_found.append({"v": round(np.mean(v_instant[start_idx:i]), 2), "dur": round(duration, 2)})
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
        
        # Centered, Scaled Video
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(res["video"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.subheader("📊 Rep Stats")
            for idx, r in enumerate(res["reps"]):
                st.markdown(f'<div class="rep-card"><b>REP {idx+1}:</b> {r["v"]} m/s | {r["dur"]}s</div>', unsafe_allow_html=True)
            
            grade = "ELITE" if res["drift"] < 2 else "STABLE" if res["drift"] < 4 else "LEAKAGE"
            st.markdown(f'<div class="form-card">⚖️ <b>FORM: {grade}</b><br>Drift: {res["drift"]} in</div>', unsafe_allow_html=True)

        with col_res2:
            st.subheader("🎯 Tactical Adjuster")
            user_rpe = st.slider("Adjust Perceived RPE", 5.0, 10.0, float(res["ai_rpe"]), 0.5)
            
            # Real-time Recalculation
            effective_reps = len(res["reps"]) + (10.0 - user_rpe)
            adj_1rm = round(weight * (36 / (37 - effective_reps)), 1) if effective_reps < 37 else weight
            
            st.markdown(f'<div class="stat-card-red">AI SUGGESTED RPE<br><h3>{res["ai_rpe"]}</h3></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="adj-card-gold"><span style="font-size:0.8em; color:#8B949E;">ADJUSTED 1RM</span><br><span style="font-size:2.2em; font-weight:900; color:#FFC107;">{adj_1rm} lbs</span></div>', unsafe_allow_html=True)

        if st.button("💾 SAVE TO CLOUD ARCHIVE", use_container_width=True):
            supabase.table("lifts").insert({"exercise": exercise, "weight": weight, "reps": len(res["reps"]), "rpe": user_rpe, "est_1rm": adj_1rm}).execute()
            st.success("Set Locked."); st.session_state.tracking_done = False; st.session_state.clicked = False; st.session_state.uploader_key += 1; st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    logs = supabase.table("lifts").select("*").order("created_at", desc=True).execute()
    if logs.data:
        for row in logs.data:
            with st.container():
                c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
                with c1: st.write(f"🏋️ **{row['exercise']}**")
                with c2: st.write(f"{row['weight']} lbs x {row['reps']}")
                with c3: st.write(f"🔥 RPE {row['rpe']}")
                with c4: st.write(f"🎯 **{row['est_1rm']}**")
                st.markdown("---")

st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
