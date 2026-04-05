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
        # Ensure ID column exists for deletion logic
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
    .stat-card-red { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #E63946; text-align: center; }
    .stat-card-green { background-color: #2D3139; padding: 20px; border-radius: 8px; border-top: 4px solid #00FF00; text-align: center; }
    /* Delete Button Styling */
    .stButton>button { border-radius: 5px; }
    div[data-testid="column"] { display: flex; align-items: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>IRON SIGHT</h1>", unsafe_allow_html=True)

# --- NAVIGATION ---
tab1, tab2 = st.tabs(["⚡ LIVE TRACK", "🗄️ TACTICAL ARCHIVE"])

# --- INITIALIZE STATE ---
if 'clicked' not in st.session_state: st.session_state.clicked = False
if 'tracking_done' not in st.session_state: st.session_state.tracking_done = False
if 'last_weight' not in st.session_state: st.session_state.last_weight = 0.0
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

with tab1:
    # (The Tracking UI remains the same as previous versions)
    # ... logic for video upload and CV tracking ...
    # Placeholder for the results after tracking:
    if st.session_state.tracking_done:
        # Example data display
        st.success("Set Analyzed.")
        if st.button("💾 SAVE TO ARCHIVE", use_container_width=True):
            # Hardcoded example values for logic demonstration
            save_to_vault(st.session_state.last_weight, 5, 8.5, 405.0)
            st.rerun()

with tab2:
    st.subheader("🗄️ TACTICAL ARCHIVE")
    df = load_vault()
    
    if df.empty:
        st.info("The Vault is currently empty.")
    else:
        # We iterate through the dataframe to create custom rows with delete buttons
        for index, row in df.iloc[::-1].iterrows(): # Show newest first
            with st.container():
                # Grid layout for each lift entry
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

        # Performance Graph
        if len(df) > 1:
            st.subheader("📈 STRENGTH TREND")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Est. 1RM'], mode='lines+markers', line=dict(color='#00FF00', width=3)))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

# --- WATERMARK ---
st.markdown('<div style="position: fixed; bottom: 15px; right: 20px; color: #595959; font-size: 0.75em; font-weight: 800; z-index: 100;">BY THE STRENGTHBENDER</div>', unsafe_allow_html=True)
