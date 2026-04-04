import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")

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
            st.info("System Initialized. Analyzing Multi-Rep Data...")

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
                bx, by, bw, bh = [int(v) for v in (box if success else bboxes[-1] if bboxes else bbox)]
                y_history.append(by + (bh//2))
                bboxes.append((bx, by, bw, bh))
                all_frames.append(frame_res)
                progress_bar.progress((i + 1) / total_frames)

            # --- MATH ENGINE ---
            smoothed_y = [np.mean(y_history[max(0, i-5):min(len(y_history), i+5)]) for i in range(len(y_history))]
            meters_per_pixel = 0.45 / bboxes[0][3]
            
            rep_starts = []
            for i in range(15, len(smoothed_y) - 15):
                if smoothed_y[i] == max(smoothed_y[i-15:i+16]) and (smoothed_y[i] - min(smoothed_y)) > 30:
                    if not rep_starts or (i - rep_starts[-1]) > (fps * 1.0):
                        rep_starts.append(i)

            rep_data = []
            for r_idx, start in enumerate(rep_starts):
                search_range = smoothed_y[start:min(start + int(fps * 3), len(smoothed_y))]
                end = start
                for j in range(5, len(search_range)):
                    if search_range[j] >= np.min(search_range[j-5:j]):
                        end = start + j - 5
                        break
                rep_y = y_history[start:end+1]
                v_list = [abs(rep_y[k-1] - rep_y[k]) * meters_per_pixel * fps for k in range(1, len(rep_y))]
                if v_list:
                    rep_data.append({"id": r_idx+1, "start": start, "end": end, "avg_v": np.mean(v_list), "peak_v": max(v_list), "duration": (end-start)/fps})

            # --- VIDEO BAKING ---
            with st.spinner("Generating Final Pro-Clip..."):
                out_frames = []
                path_points = []
                for i in range(len(all_frames)):
                    frame_draw = all_frames[i].copy()
                    bx, by, bw, bh = bboxes[i]
                    active_rep = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                    
                    path_points.append((bx + bw//2, by + bh//2))
                    if len(path_points) > 1:
                        for j in range(max(1, i-90), len(path_points)):
                            cv2.line(frame_draw, path_points[j-1], path_points[j], (255, 75, 173), 2, cv2.LINE_AA)

                    # Overlay logic
                    cv2.rectangle(frame_draw, (0, 0), (new_width, 80), (0, 0, 0), -1)
                    if active_rep:
                        elapsed = (i - active_rep['start']) / fps
                        cv2.putText(frame_draw, f"REP {active_rep['id']} | {elapsed:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame_draw, f"VEL: {active_rep['avg_v']:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame_draw, "TheStrengthBenderSystem", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                    
                    out_frames.append(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))

                # --- GENERATE END CARD ---
                if rep_data:
                    # Create a black frame for the summary
                    summary_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    cv2.putText(summary_frame, "THE STRENGTH BENDER", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 75, 173), 2)
                    cv2.putText(summary_frame, "SYSTEM STATS", (120, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.line(summary_frame, (40, 110), (new_width - 40, 110), (100, 100, 100), 1)

                    y_offset = 160
                    for r in rep_data:
                        # FIXED LINE BELOW: moved 's' outside of the formatting bracket
                        cv2.putText(summary_frame, f"REP {r['id']}: {r['duration']:.2f}s | {r['avg_v']:.2f} m/s", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(summary_frame, f"PEAK: {r['peak_v']:.2f} m/s", (60, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        y_offset += 70

                    # Add the summary frame for 3 seconds (fps * 3)
                    for _ in range(int(fps * 3)):
                        out_frames.append(summary_frame)

                final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                imageio.mimsave(final_path, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                st.video(final_path)
