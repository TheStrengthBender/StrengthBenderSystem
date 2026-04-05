import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from supabase import create_client, Client
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CLOUD CONNECTION ---
# These connect directly to your 'ohnzahpomkezpogfhcvj' project
SUPABASE_URL = "https://ohnzahpomkezpogfhcvj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obnphaHBvbWtlenBvZ2ZoY3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU0MjAyMjcsImV4cCI6MjA5MDk5NjIyN30.d0kvd-bEKWeVsnmXf9UvWyQsCUuSYOubNgnlUpt-vqw"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="IRON SIGHT", page_icon="🎯", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #1A1C20; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF !important; font-weight: 900; text-align: center; text-transform: uppercase; margin-bottom: 0px;}
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    .pr-banner { background-color: rgba(0, 255, 0, 0.1); border: 1px solid #00FF00; padding: 15px; border-radius: 8px; text-align: center; color: #00FF00; font-weight: bold; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E; margin-bottom: 30px; letter-spacing: 2px;'>TACTICAL VELOCITY TRACKER</p>", unsafe_allow_html=True)

# --- APP SETTINGS ---
EXERCISES = ["Squat", "Bench Press", "Deadlift", "Overhead Press", "Barbell Row"]

# --- INITIALIZE STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'last_weight' not in st.session_state: st.session_state.last_weight = 0.0
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

tab1, tab2 = st.tabs(["⚡ LIVE TRACK", "🗄️ TACTICAL ARCHIVE"])

with tab1:
    col_ex, col_w = st.columns(2)
    with col_ex:
        exercise = st.selectbox("Select Exercise", EXERCISES)
    with col_w:
        weight = st.number_input("Weight (lbs)", min_value=0.0, step=5.0, value=st.session_state.last_weight)
        st.session_state.last_weight = weight

    if not st.session_state.tracking_done:
        uploaded_file = st.file_uploader("Upload Set", type=["mp4", "mov"], key=f"uploader_{st.session_state.uploader_key}")

        if uploaded_file and weight > 0:
            tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
            with open(tpath, "wb") as f: f.write(uploaded_file.read())
            cap = cv2.VideoCapture(tpath)
            ret, first_frame = cap.read()
            
            if ret:
                display_w = 320
                first_frame_res = cv2.resize(first_frame, (display_w, int(display_w*(first_frame.shape[0]/first_frame.shape[1]))))
                
                if not st.session_state.clicked:
                    st.markdown("### 🎯 Lock the Target")
                    value = streamlit_image_coordinates(cv2.cvtColor(first_frame_res, cv2.COLOR_BGR2RGB), key="clicker")
                    if value:
                        st.session_state.clicked = True
                        # SIMULATED PHYSICS for testing cloud save
                        st.session_state.rep_data = {"reps": 1, "rpe": 8.0, "1rm": weight * 1.15}
                        st.session_state.tracking_done = True
                        st.rerun()

    if st.session_state.tracking_done:
        data = st.session_state.rep_data
        
        # PR CHECKER
        best_query = supabase.table("lifts").select("est_1rm").eq("exercise", exercise).order("est_1rm", desc=True).limit(1).execute()
        if best_query.data and data["1rm"] > best_query.data[0]["est_1rm"]:
            st.markdown('<div class="pr-banner">🏆 NEW PERSONAL RECORD DETECTED!</div>', unsafe_allow_html=True)
            st.balloons()

        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="stat-card-red">AI RPE<br><h2>{data["rpe"]}</h2></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-card-green">EST 1RM<br><h2>{data["1rm"]:.1f}</h2></div>', unsafe_allow_html=True)

        if st.button("💾 SAVE TO CLOUD ARCHIVE", use_container_width=True):
            entry = {
                "exercise": exercise,
                "weight": weight,
                "reps": data["reps"],
                "rpe": data["rpe"],
                "est_1rm": data["1rm"]
            }
            # THIS SENDS THE DATA TO SUPABASE
            supabase.table("lifts").insert(entry).execute()
            st.success(f"{exercise} locked into the cloud.")
            st.session_state.tracking_done = False
            st.session_state.clicked = False
            st.session_state.uploader_key += 1
            st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    # Fetch all data from Supabase
    logs = supabase.table("lifts").select("*").order("created_at", desc=True).execute()
    
    if not logs.data:
        st.info("The Cloud Vault is empty. Log your first set to see history.")
    else:
        for row in logs.data:
            with st.container():
                c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
                with c1: st.write(f"🏋️ **{row['exercise']}**")
                with c2: st.write(f"{row['weight']} lbs")
                with c3: st.write(f"RPE {row['rpe']}")
                with c4: st.write(f"🎯 **{row['est_1rm']:.1f}**")
                st.markdown("---")

# --- WATERMARK ---
st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
