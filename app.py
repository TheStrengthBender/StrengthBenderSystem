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
            st.info("System Initialized. Filtering out setup noise...")

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

            # --- ROBUST REP DETECTION ---
            # Increase smoothing to ignore tremors
            smoothed_y = [np.mean(y_history[max(0, i-7):min(len(y_history), i+7)]) for i in range(len(y_history))]
            meters_per_pixel = 0.45 / bboxes[0][3]
            
            rep_starts = []
            for i in range(20, len(smoothed_y) - 20):
                # Only identify a 'bottom' if it's a significant dip compared to the rest of the lift
                if smoothed_y[i] == max(smoothed_y[i-20:i+21]):
                    # Calculate depth: how high did it go from the baseline?
                    if (smoothed_y[i] - min(smoothed_y)) > 40: # Minimum 40 pixel movement to count
                        if not rep_starts or (i - rep_starts[-1]) > (fps * 1.5):
                            rep_starts.append(i)

            rep_data = []
            for r_idx, start in enumerate(rep_starts):
                search_range = smoothed_y[start:min(start + int(fps * 4), len(smoothed_y))]
                end = start
                # Look for the peak of the pull
                for j in range(5, len(search_range)):
                    if search_range[j] <= min(search_range) or search_range[j] >= search_range[j-1]:
                        if j > 10: # Min duration check
                            end = start + j
                            break
                
                rep_y = y_history[start:end+1]
                # Filter velocity spikes
                v_raw = [abs(rep_y[k-1] - rep_y[k]) * meters_per_pixel * fps for k in range(1, len(rep_y))]
                v_smooth = [np.mean(v_raw[max(0, m-2):min(len(v_raw), m+2)]) for m in range(len(v_raw))]
                
                if v_smooth and (max(rep_y) - min(rep_y)) * meters_per_pixel > 0.15: # Must move > 15cm
                    rep_data.append({
                        "id": len(rep_data) + 1, 
                        "start": start, "end": end, 
                        "avg_v": np.mean(v_smooth), 
                        "peak_v": max(v_smooth), 
                        "duration": (end-start)/fps
                    })

            # --- DASHBOARD & BAKING ---
            col1, col2 = st.columns([3, 1])
            with col2:
                st.subheader("📊 Set Stats")
                for r in rep_data:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} avg | {r["peak_v"]:.2f} peak<br>Time: {r["duration"]:.2f}s</div>', unsafe_allow_html=True)

            with col1:
                with st.spinner("Baking Pro-Video..."):
                    out_frames, path_points = [], []
                    for i in range(len(all_frames)):
                        frame_draw = all_frames[i].copy()
                        bx, by, bw, bh = bboxes[i]
                        active_rep = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                        
                        path_points.append((bx + bw//2, by + bh//2))
                        if len(path_points) > 1:
                            for j in range(max(1, i-120), len(path_points)):
                                cv2.line(frame_draw, path_points[j-1], path_points[j], (255, 75, 173), 2, cv2.LINE_AA)

                        cv2.rectangle(frame_draw, (0, 0), (new_width, 80), (0, 0, 0), -1)
                        if active_rep:
                            elapsed = (i - active_rep['start']) / fps
                            cv2.putText(frame_draw, f"REP {active_rep['id']} | {elapsed:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame_draw, "STRENGTH BENDER - READY", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        
                        out_frames.append(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))

                    # Final Summary Slide
                    if rep_data:
                        summary_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                        cv2.putText(summary_frame, "THE STRENGTH BENDER", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 75, 173), 2)
                        y_offset = 140
                        for r in rep_data:
                            cv2.putText(summary_frame, f"REP {r['id']}: {r['duration']:.2f}s | {r['avg_v']:.2f}m/s", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_offset += 50
                        for _ in range(int(fps * 4)): out_frames.append(summary_frame)

                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    imageio.mimsave(final_path, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(final_path)
