import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
import pandas as pd
from datetime import datetime
from streamlit_image_coordinates import streamlit_image_coordinates
import plotly.graph_objects as go

st.set_page_config(page_title="IRON SIGHT", page_icon="🎯", layout="wide")

# --- DATABASE LOGIC ---
DB_FILE = "vault.csv"

def load_vault():
    if os.path.isfile(DB_FILE):
        df = pd.read_csv(DB_FILE)
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(len(df)))
            df.to_csv(DB_FILE, index=False)
        return df
    return pd.DataFrame(columns=["ID", "Date", "Weight", "Reps", "RPE", "Est. 1RM"])

def save_to_vault(weight, reps, rpe, est_1rm):
    df = load_vault()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_id = df['ID'].max() + 1 if not df.empty else 0
    new_entry = pd.DataFrame([[new_id, now, weight, reps, rpe, est_1rm]], 
                            columns=["ID", "Date", "Weight", "Reps", "RPE", "Est. 1RM"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def delete_from_vault(row_id):
    df = load_vault()
    df = df[df.ID != row_id]
    df.to_csv(DB_FILE, index=False)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF !important; font-weight: 900; text-align: center; text-transform: uppercase; margin-bottom: 0px;}
    .rep-card { background-color: #2D3139; padding: 15px; border-radius: 8px; border-left: 4px solid #E63946; margin-bottom: 10px; color: white; }
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { color: #8B949E; }
    .stTabs [aria-selected="true"] { color: white !important; border-bottom-color: #E63946 !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E; margin-bottom: 30px; letter-spacing: 2px;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'last_weight' not in st.session_state: st.session_state.last_weight = 0.0
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'saved_this_set' not in st.session_state: st.session_state.saved_this_set = False

# --- NAVIGATION ---
tab1, tab2 = st.tabs(["⚡ LIVE TRACK", "🗄️ TACTICAL ARCHIVE"])

with tab1:
    if not st.session_state.tracking_done:
        col_w, col_u = st.columns([1, 2])
        with col_w:
            weight_in = st.number_input("Weight on Bar (lbs)", min_value=0.0, value=st.session_state.last_weight, step=5.0)
            st.session_state.last_weight = weight_in
        with col_u:
            uploaded_file = st.file_uploader("Upload MP4/MOV", type=["mp4", "mov"], key=f"uploader_{st.session_state.uploader_key}")

        if uploaded_file and st.session_state.last_weight > 0:
            tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
            with open(tpath, "wb") as f: f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            
            if ret:
                display_w = 320
                frame_rgb = cv2.cvtColor(cv2.resize(first_frame, (640, int(640*(first_frame.shape[0]/first_frame.shape[1])))), cv2.COLOR_BGR2RGB)
                first_frame_res = cv2.resize(frame_rgb, (display_w, int(display_w*(first_frame.shape[0]/first_frame.shape[1]))))
                
                if not st.session_state.clicked:
                    st.markdown("### 🎯 Lock the Target")
                    value = streamlit_image_coordinates(first_frame_res, key="clicker")
                    if value:
                        st.session_state.coords = (value['x']*(640/display_w), value['y']*(640/display_w))
                        st.session_state.clicked = True
                        st.rerun()

                if st.session_state.clicked:
                    # Physics Tracking (CSRT)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(cv2.resize(first_frame, (640, int(640*(first_frame.shape[0]/first_frame.shape[1])))), 
                                (int(st.session_state.coords[0]-20), int(st.session_state.coords[1]-20), 40, 40))
                    
                    y_hist, bboxes = [], []
                    progress = st.progress(0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    for i in range(total_frames):
                        ret, f = cap.read()
                        if not ret: break
                        f_rs = cv2.resize(f, (640, int(640*(f.shape[0]/f.shape[1]))))
                        ok, box = tracker.update(f_rs)
                        y_hist.append(box[1]+box[3]/2)
                        bboxes.append(box)
                        progress.progress((i+1)/total_frames)
                    
                    # Logic
                    m_per_px = 0.45 / bboxes[0][3]
                    v_instant = [(y_hist[j-1]-y_hist[j])*m_per_px*fps if j>0 else 0 for j in range(len(y_hist))]
                    v_smooth = [np.mean(v_instant[max(0, x-3):min(len(v_instant), x+3)]) for x in range(len(v_instant))]
                    
                    # Estimate RPE/1RM
                    max_v = max(v_smooth)
                    est_rpe = 8.0 if max_v < 0.5 else 7.0 if max_v < 0.7 else 6.0
                    est_1rm = st.session_state.last_weight * (1.1 if est_rpe == 8 else 1.2)
                    
                    st.session_state.rep_data = {"reps": 1, "rpe": est_rpe, "1rm": est_1rm}
                    st.session_state.tracking_done = True
                    st.rerun()

    if st.session_state.tracking_done:
        data = st.session_state.rep_data
        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="stat-card-red">AI RPE<br><h2>{data["rpe"]}</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-card-green">EST 1RM<br><h2>{data["1rm"]:.1f}</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.saved_this_set:
            if st.button("💾 SAVE TO ARCHIVE", use_container_width=True):
                save_to_vault(st.session_state.last_weight, data["reps"], data["rpe"], data["1rm"])
                st.session_state.saved_this_set = True
                st.rerun()
        
        if st.button("🔄 TRACK NEW SET", use_container_width=True):
            st.session_state.clicked = False; st.session_state.tracking_done = False; st.session_state.saved_this_set = False; st.session_state.uploader_key += 1; st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    df_vault = load_vault()
    
    if df_vault.empty:
        st.info("The Vault is currently empty.")
    else:
        for idx, row in df_vault.iloc[::-1].iterrows():
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 0.5])
                with c1: st.write(f"📅 **{row['Date']}**")
                with c2: st.write(f"🏋️ {row['Weight']} lbs")
                with c3: st.write(f"🔥 RPE {row['RPE']}")
                with c4: st.write(f"🎯 **{row['Est. 1RM']:.1f}**")
                with c5:
                    if st.button("🗑️", key=f"del_{row['ID']}"):
                        delete_from_vault(row['ID'])
                        st.rerun()
                st.markdown("---")

# --- WATERMARK ---
st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
