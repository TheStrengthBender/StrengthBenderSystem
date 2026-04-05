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

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF !important; font-weight: 900; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px;}
    .rep-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #E63946; margin-bottom: 10px; color: white; }
    .form-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #8B949E; margin-bottom: 10px; color: white; }
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    .est-card-gold { background-color: #2D3139; padding: 20px; border-radius: 8px; border-left: 5px solid #FFC107; margin-top: 15px; color: white; text-align: center; }
    .pr-banner { background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00FF00; padding: 15px; border-radius: 8px; text-align: center; color: #00FF00; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
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
                            x_hist.append(bx + bw//2); y_hist.append(by + bh//2); bboxes.append(box)
                            if len(x_hist) > 1:
                                for j in range(1, len(x_hist)):
                                    color = (0, 255, 0) if y_hist[j] < y_hist[j-1] else (255, 255, 255)
                                    cv2.line(frame, (x_hist[j-1], y_hist[j-1]), (x_hist[j], y_hist[j]), color, 3)
                        frames_out.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                        progress.progress((i + 1) / total_frames)
                    
                    m_per_px = 0.45 / bboxes[0][3]
                    v_instant = [(y_hist[j-1]-y_hist[j])*m_per_px*fps if j>0 else 0 for j in range(len(y_hist))]
                    avg_v = np.mean([v for v in v_instant if v > 0.15]) if any(v > 0.15 for v in v_instant) else 0
                    drift_in = (max(x_hist) - min(x_hist)) * m_per_px * 39.37
                    
                    # Initial AI Estimate
                    est_rpe = max(6.0, min(10.0, 10.5 - (avg_v * 4)))
                    final_rpe = round(est_rpe * 2) / 2
                    
                    out_path = os.path.join(tempfile.gettempdir(), "tracked.mp4")
                    imageio.mimsave(out_path, frames_out, fps=fps, codec='libx264')
                    
                    st.session_state.rep_data = {"v": round(avg_v, 2), "ai_rpe": final_rpe, "video": out_path, "drift": round(drift_in, 1)}
                    st.session_state.tracking_done = True; st.rerun()

    if st.session_state.tracking_done:
        res = st.session_state.rep_data
        st.video(res["video"])
        
        # --- ADJUSTABLE RPE SLIDER ---
        st.markdown("### 🎛️ Tactical Adjustment")
        user_rpe = st.slider("Adjust Perceived RPE", 5.0, 10.0, float(res["ai_rpe"]), 0.5)
        
        # --- REVERSE MATH ENGINE ---
        rir = 10.0 - user_rpe
        effective_reps = 1 + rir # Assuming 1 rep tracked
        current_1rm = round(weight * (36 / (37 - effective_reps)), 1)
        
        # PR Check
        best_query = supabase.table("lifts").select("est_1rm").eq("exercise", exercise).order("est_1rm", desc=True).limit(1).execute()
        if best_query.data and current_1rm > best_query.data[0]["est_1rm"]:
            st.markdown('<div class="pr-banner">🏆 NEW PERSONAL RECORD DETECTED!</div>', unsafe_allow_html=True)
            st.balloons()

        st.markdown(f'<div class="rep-card"><b>REP 1:</b> {res["v"]} m/s</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="stat-card-red"><span style="color:#8B949E; font-size:0.8em;">FINAL RPE</span><br><span style="font-size:2em; font-weight:900;">{user_rpe}</span></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-card-green"><span style="color:#8B949E; font-size:0.8em;">EST 1RM</span><br><span style="font-size:2em; font-weight:900;">{current_1rm}</span></div>', unsafe_allow_html=True)

        if st.button("💾 SAVE TO CLOUD ARCHIVE", use_container_width=True):
            entry = {"exercise": exercise, "weight": weight, "reps": 1, "rpe": user_rpe, "est_1rm": current_1rm}
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
                with c3: st.write(f"🔥 RPE {row['rpe']}")
                with c4: st.write(f"🎯 **{row['est_1rm']}**")
                st.markdown("---")

st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
