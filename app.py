import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# --- THE STRENGTH BENDER THEME ---
st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")

uploaded_file = st.file_uploader("Upload Set (MP4 or MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mov') 
    tfile.write(uploaded_file.read())
    tfile.flush()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if ret:
        h, w = first_frame.shape[:2]
        new_width = 400
        new_height = int(new_width * (h / w))
        if new_height % 2 != 0: new_height += 1
        first_frame_resized = cv2.resize(first_frame, (new_width, new_height))

        st.markdown("### 🎯 Step 1: Click on a point of the barbell")
        value = streamlit_image_coordinates(cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2RGB), key="clicker")

        if value is not None:
            cx, cy = value['x'], value['y']
            bbox = (cx - 25, cy - 25, 50, 50)
            st.info("System Initialized. Analyzing Data...")

            tracker = cv2.TrackerCSRT_create() 
            tracker.init(first_frame_resized, bbox)

            y_history, bboxes, all_frames = [], [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progress_bar = st.progress(0)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                frame_res = cv2.resize(frame, (new_width, new_height))
                success, box = tracker.update(frame_res)
                if success:
                    bx, by, bw, bh = [int(v) for v in box]
                    y_history.append(by + (bh//2))
                    bboxes.append((bx, by, bw, bh))
                else:
                    y_history.append(y_history[-1] if y_history else cy)
                    bboxes.append(bboxes[-1] if bboxes else bbox)
                all_frames.append(frame_res)
                progress_bar.progress((i + 1) / total_frames)

            # --- ROBUST REP DETECTION ---
            smoothed_y = [np.mean(y_history[max(0, i-5):min(len(y_history), i+5)]) for i in range(len(y_history))]
            meters_px = 0.45 / bboxes[0][3]
            reps = []
            
            for i in range(15, len(smoothed_y) - 15):
                # Detect the bottom of the rep
                if smoothed_y[i] == max(smoothed_y[i-15:i+16]) and (smoothed_y[i] - min(smoothed_y)) > 30:
                    if not reps or (i - reps[-1]['start']) > fps:
                        # Precision Snap-Back
                        end = i
                        for j in range(i + 2, min(i + int(fps * 3), len(smoothed_y))):
                            if smoothed_y[j] >= smoothed_y[j-1] - 0.5:
                                end = j
                                break
                        
                        # Check if it was a real rep (moved > 20cm)
                        segment = y_history[i:end+1]
                        if (max(segment) - min(segment)) * meters_px > 0.15:
                            v_raw = [abs(y_history[k-1] - y_history[k]) * meters_px * fps for k in range(i+1, end+1)]
                            reps.append({
                                "id": len(reps) + 1, "start": i, "end": end, 
                                "avg": np.mean(v_raw), "peak": max(v_raw), "dur": (end-i)/fps
                            })

            # --- SINGLE BAKING PROCESS ---
            col1, col2 = st.columns([3, 1])
            with col1:
                with st.spinner("Baking One Final Video..."):
                    out_frames = []
                    path_points = []
                    for i in range(len(all_frames)):
                        frame_draw = all_frames[i].copy()
                        bx, by, bw, bh = bboxes[i]
                        active = next((r for r in reps if r['start'] <= i <= r['end']), None)
                        
                        path_points.append((bx + bw//2, by + bh//2))
                        if len(path_points) > 1:
                            for j in range(max(1, i-90), len(path_points)):
                                cv2.line(frame_draw, path_points[j-1], path_points[j], (255, 75, 173), 2, cv2.LINE_AA)

                        cv2.rectangle(frame_draw, (0, 0), (new_width, 65), (0, 0, 0), -1)
                        if active:
                            cv2.putText(frame_draw, f"REP {active['id']} | {(i-active['start'])/fps:.2f}s", (10, 30), 2, 0.6, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame_draw, "STRENGTH BENDER READY", (10, 40), 2, 0.5, (150, 150, 150), 1)
                        
                        out_frames.append(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))

                    # ADD THE END CARD
                    if reps:
                        card = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                        cv2.putText(card, "THE STRENGTH BENDER", (30, 60), 2, 0.8, (255, 75, 173), 2)
                        y_off = 130
                        for r in reps:
                            cv2.putText(card, f"R{r['id']}: {r['dur']:.2f}s | {r['avg']:.2f}m/s", (30, y_off), 2, 0.5, (255, 255, 255), 1)
                            y_off += 50
                        for _ in range(int(fps * 4)): out_frames.append(card)

                    f_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    imageio.mimsave(f_path, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(f_path)

            with col2:
                st.subheader("📊 Results")
                for r in reps:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg"]:.2f} m/s avg<br>{r["dur"]:.2f}s</div>', unsafe_allow_html=True)
