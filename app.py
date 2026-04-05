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

# --- UI REFINEMENT ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .video-container { max-width: 550px; margin: 0 auto; border-radius: 10px; overflow: hidden; border: 1px solid #30363d; margin-bottom: 20px;}
    .card { background-color: #161B22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; margin-bottom: 10px; }
    .stat-label { color: #8B949E; font-size: 0.8em; text-transform: uppercase; font-weight: 700; display: block; }
    .stat-value { font-size: 2.2em; font-weight: 900; color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; margin-bottom: 25px;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

# --- STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

EXERCISES = ["Squat", "Deadlift", "Bench Press", "Overhead Press"]
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
            
            frame_step = 1 if total_frames < 300 else 2
            
            ret, first_frame = cap.read()
            if ret:
                h, w = first_frame.shape[:2]
                display_w = 480
                display_h = int(display_w * (h / w))
                first_frame_res = cv2.resize(first_frame, (display_w, display_h))
                
                if not st.session_state.clicked:
                    st.markdown("### 🎯 Set Reference Point")
                    value = streamlit_image_coordinates(cv2.cvtColor(first_frame_res, cv2.COLOR_BGR2RGB), key="clicker")
                    if value:
                        st.session_state.coords = (value['x'] * (w/display_w), value['y'] * (h/display_h))
                        st.session_state.clicked = True; st.rerun()

                if st.session_state.clicked:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(first_frame, (int(st.session_state.coords[0]-30), int(st.session_state.coords[1]-30), 60, 60))
                    
                    y_hist, x_hist, bboxes, frames_out = [], [], [], []
                    progress = st.progress(0)
                    
                    for i in range(0, total_frames, frame_step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if not ret: break
                        
                        ok, box = tracker.update(frame)
                        if ok:
                            cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
                            x_hist.append(cx); y_hist.append(cy); bboxes.append(box)
                            
                            # --- SMART PATH VISUALS (Restored) ---
                            # Only draws the path of the bar. Green = Ascending, White = Descending/Walkout
                            if len(x_hist) > 1:
                                for j in range(1, len(x_hist)):
                                    color = (0, 255, 0) if y_hist[j] < y_hist[j-1] else (255, 255, 255)
                                    cv2.line(frame, (int(x_hist[j-1]), int(y_hist[j-1])), (int(x_hist[j]), int(y_hist[j])), color, 3)
                        
                        frames_out.append(cv2.cvtColor(cv2.resize(frame, (display_w, display_h)), cv2.COLOR_BGR2RGB))
                        progress.progress(i / total_frames)
                    
                    # --- NOISE-ELIMINATING ANALYTICS ---
                    m_per_px = 0.45 / bboxes[0][3]
                    v_instant = [(y_hist[j-1]-y_hist[j])*m_per_px*(fps/frame_step) if j>0 else 0 for j in range(len(y_hist))]
                    
                    # Smooth the velocity to ignore micro-shakes
                    v_smooth = [np.mean(v_instant[max(0, k-3):min(len(v_instant), k+3)]) for k in range(len(v_instant))]
                    
                    reps = []
                    is_moving = False; start_idx = 0
                    for i, v in enumerate(v_smooth):
                        if not is_moving and v > 0.1: # Threshold to start rep
                            is_moving, start_idx = True, i
                        elif is_moving and v < 0: # Threshold to end rep
                            duration = (i - start_idx) / (fps/frame_step)
                            peak_v = max(v_smooth[start_idx:i])
                            
                            # STRICT FILTER: Must take >0.5s AND hit a peak speed >0.15m/s (ignores unracks/shakes)
                            if duration > 0.5 and peak_v > 0.15:
                                # Isolate horizontal drift to ONLY this specific rep (ignores walkout)
                                rep_x = x_hist[start_idx:i]
                                rep_drift_in = (max(rep_x) - min(rep_x)) * m_per_px * 39.37
                                
                                reps.append({
                                    "v": round(np.mean(v_smooth[start_idx:i]), 2), 
                                    "dur": round(duration, 2),
                                    "drift": round(rep_drift_in, 1)
                                })
                            is_moving = False

                    # Final Math based ONLY on validated reps
                    if reps:
                        final_drift = max([r["drift"] for r in reps])
                        avg_v = reps[0]["v"] # Uses first valid rep for baseline RPE calculation
                        est_rpe = round(max(6.0, min(10.0, 11.0 - (avg_v * 5))) * 2) / 2
                    else:
                        # Fallback if the strict filter accidentally wipes out a super slow grinder
                        final_drift = 0.0
                        est_rpe = 10.0

                    out_path = os.path.join(tempfile.gettempdir(), "tracked.mp4")
                    imageio.mimsave(out_path, frames_out, fps=fps/frame_step, codec='libx264')
                    
                    st.session_state.rep_data = {"reps": reps, "ai_rpe": est_rpe, "video": out_path, "drift": final_drift}
                    st.session_state.tracking_done = True; st.rerun()

    if st.session_state.tracking_done:
        res = st.session_state.rep_data
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(res["video"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Performance Data")
        if not res["reps"]:
            st.error("No valid reps detected. Ensure clear bar path and clean lighting.")
        else:
            for idx, r in enumerate(res["reps"]):
                st.markdown(f'<div class="card" style="border-left: 4px solid #FF1E56; padding: 10px 20px;">REP {idx+1}: {r["v"]} m/s | {r["dur"]}s</div>', unsafe_allow_html=True)
            
            grade = "STABLE" if res["drift"] < 4.5 else "LEAKAGE"
            st.markdown(f'<div class="card" style="border-left: 4px solid #00D2FF; padding: 10px 20px;">FORM GRADE: {grade} | Drift: {res["drift"]} in</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="card" style="text-align: center;"><span class="stat-label">AI SUGGESTED RPE</span><div class="stat-value">{res["ai_rpe"]}</div></div>', unsafe_allow_html=True)
            
            user_rpe = st.slider("Manual Override", 5.0, 10.0, float(res["ai_rpe"]), 0.5)
            adj_1rm = round(weight * (36 / (37 - (1 + (10 - user_rpe)))), 1)
            
            st.markdown(f'<div class="card" style="text-align: center; border-top: 4px solid #00D2FF;"><span class="stat-label">AI 1RM EST</span><div class="stat-value" style="color:#00D2FF;">{adj_1rm}</div></div>', unsafe_allow_html=True)

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
