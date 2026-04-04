import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# --- THE STRENGTH BENDER THEME v2.0 ---
st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    button[kind="primary"], .stButton > button {
        background-color: #FF4BAD !important;
        color: white !important;
        border-radius: 8px !important;
    }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    h3 { color: #FF4BAD !important; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")
st.write("Advanced Multi-Rep Volumetric Analysis")

uploaded_file = st.file_uploader("Upload Set (MP4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        new_width = 400
        new_height = int(new_width * (h / w))
        if new_height % 2 != 0: new_height += 1
        first_frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

        st.markdown("### 🎯 Step 1: Click on a point of the barbell")
        value = streamlit_image_coordinates(first_frame_resized, key="clicker")

        if value is not None:
            cx, cy = value['x'], value['y']
            bbox = (cx - 25, cy - 25, 50, 50)
            st.info("System Initialized. Analyzing Volume...")

            tracker = cv2.TrackerCSRT_create() 
            tracker.init(cv2.cvtColor(first_frame_resized, cv2.COLOR_RGB2BGR), bbox)

            y_history, bboxes, all_frames = [], [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progress_bar = st.progress(0)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                frame_res = cv2.resize(frame, (new_width, new_height))
                success, box = tracker.update(frame_res)
                bx, by, bw, bh = [int(v) for v in (box if success else bboxes[-1] if bboxes else bbox)]
                y_history.append(by + (bh//2))
                bboxes.append((bx, by, bw, bh))
                all_frames.append(cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB))
                progress_bar.progress((i + 1) / total_frames)

            # --- MULTI-REP DETECTION LOGIC ---
            smoothed_y = [np.mean(y_history[max(0, i-5):min(len(y_history), i+5)]) for i in range(len(y_history))]
            
            # Find all local maximums (bottom of reps)
            rep_starts = []
            for i in range(10, len(smoothed_y) - 10):
                if smoothed_y[i] == max(smoothed_y[i-10:i+11]) and (smoothed_y[i] - min(smoothed_y)) > 20:
                    if not rep_starts or (i - rep_starts[-1]) > (fps * 0.5): # Minimum 0.5s between reps
                        rep_starts.append(i)

            rep_data = []
            meters_per_pixel = 0.45 / bboxes[0][3]

            # Analyze each detected rep
            for r_idx, start in enumerate(rep_starts):
                # Find lockout (next point where velocity stabilizes or bar is high)
                search_range = smoothed_y[start:start + int(fps * 3)] # Max 3s ascent
                lockout_offset = search_range.index(min(search_range))
                end = start + lockout_offset
                
                # Calculate metrics
                rep_y = y_history[start:end+1]
                v_list = [abs(rep_y[j-1] - rep_y[j]) * meters_per_pixel * fps for j in range(1, len(rep_y))]
                
                if v_list:
                    rep_data.append({
                        "id": r_idx + 1,
                        "start": start,
                        "end": end,
                        "avg_v": np.mean(v_list),
                        "peak_v": max(v_list),
                        "duration": (end - start) / fps
                    })

            # --- DISPLAY RESULTS ---
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("📊 Set Summary")
                for rep in rep_data:
                    st.markdown(f"""
                    <div class="rep-card">
                        <b>REP {rep['id']}</b><br>
                        Avg: {rep['avg_v']:.2f} m/s | Peak: {rep['peak_v']:.2f} m/s<br>
                        Time: {rep['duration']:.2f}s
                    </div>
                    """, unsafe_allow_html=True)

            with col1:
                # Bake video for just the first few reps to show the tech
                with st.spinner("Baking Volumetric Video..."):
                    out_frames = []
                    for i in range(len(all_frames)):
                        frame_draw = all_frames[i].copy()
                        bx, by, bw, bh = bboxes[i]
                        
                        # Find if we are currently in a rep
                        active_rep = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                        
                        # Draw Path & Dashboard
                        color = (255, 255, 0) if active_rep else (173, 75, 255)
                        cv2.rectangle(frame_draw, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                        
                        cv2.rectangle(frame_draw, (0, 0), (new_width, 50), (0, 0, 0), -1)
                        status = f"REP {active_rep['id']} ACTIVE" if active_rep else "WAITING..."
                        cv2.putText(frame_draw, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        out_frames.append(frame_draw)

                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    imageio.mimsave(final_path, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(final_path)