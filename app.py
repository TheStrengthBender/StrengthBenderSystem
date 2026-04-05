import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# --- THE STRENGTH BENDER THEME ---
st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="S/B", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; color: white; }
    button[kind="primary"] { background-color: #FF4BAD !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")

uploaded_file = st.file_uploader("Upload Set (MP4 or MOV)", type=["mp4", "mov"])

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
        # Force RGB for Mobile Safari compatibility
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        new_width = 400
        new_height = int(new_width * (h / w))
        if new_height % 2 != 0: new_height += 1
        first_frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

        st.markdown("### 🎯 Step 1: Click on the Barbell")
        # Unique key helps Safari re-render if the file changes
        value = streamlit_image_coordinates(first_frame_resized, key="click_v7")

        if value is not None:
            cx, cy = value['x'], value['y']
            bbox = (cx - 25, cy - 25, 50, 50)
            st.info("System Initialized. Analyzing Motion...")

            tracker = cv2.TrackerCSRT_create() 
            tracker.init(cv2.resize(first_frame, (new_width, new_height)), bbox)

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

            # --- UNIVERSAL HYBRID REP DETECTION ---
            smoothed_y = [np.mean(y_history[max(0, i-7):min(len(y_history), i+7)]) for i in range(len(y_history))]
            meters_per_pixel = 0.45 / bboxes[0][3]
            
            # Find all potential 'starts' (bottom of squat or floor for deadlift)
            potential_starts = []
            for i in range(20, len(smoothed_y) - 20):
                if smoothed_y[i] == max(smoothed_y[i-20:i+21]):
                    # Check if it's a deep enough position
                    if (smoothed_y[i] - min(smoothed_y)) > 35: 
                        if not potential_starts or (i - potential_starts[-1]) > (fps * 1.0):
                            potential_starts.append(i)

            rep_data = []
            for start in potential_starts:
                search_range = smoothed_y[start:min(start + int(fps * 3), len(smoothed_y))]
                end = start
                for j in range(2, len(search_range)):
                    # SNAP: Instant lockout detection
                    if search_range[j] >= search_range[j-1]: 
                        end = start + j
                        break
                
                # Validation: Displacement > 20cm
                if (smoothed_y[start] - smoothed_y[end]) * meters_per_pixel > 0.20:
                    rep_y = y_history[start:end+1]
                    v_raw = [abs(rep_y[k-1] - rep_y[k]) * meters_per_pixel * fps for k in range(1, len(rep_y))]
                    
                    rep_data.append({
                        "id": len(rep_data) + 1, 
                        "start": start, "end": end, 
                        "avg_v": np.mean(v_raw), 
                        "peak_v": max(v_raw), 
                        "duration": (end - start) / fps
                    })

            # --- OUTPUT ---
            col1, col2 = st.columns([3, 1])
            with col2:
                st.subheader("📊 Stats")
                for r in rep_data:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f}m/s | {r["duration"]:.2f}s</div>', unsafe_allow_html=True)

            with col1:
                with st.spinner("Baking Final Analytics..."):
                    out_frames, path_points = [], []
                    for i in range(len(all_frames)):
                        frame_draw = all_frames[i].copy()
                        bx, by, bw, bh = bboxes[i]
                        active_rep = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                        
                        path_points.append((bx + bw//2, by + bh//2))
                        if len(path_points) > 1:
                            # Show longer path trail
                            for j in range(max(1, i-150), len(path_points)):
                                cv2.line(frame_draw, path_points[j-1], path_points[j], (255, 75, 173), 2, cv2.LINE_AA)

                        cv2.rectangle(frame_draw, (0, 0), (new_width, 70), (0, 0, 0), -1)
                        if active_rep:
                            curr_time = (i - active_rep['start']) / fps
                            cv2.putText(frame_draw, f"REP {active_rep['id']} | {curr_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            cv2.putText(frame_draw, f"VEL: {active_rep['avg_v']:.2f}m/s", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.putText(frame_draw, "STRENGTH BENDER READY", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        
                        out_frames.append(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))

                    # --- END CARD ---
                    if rep_data:
                        card = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                        cv2.putText(card, "STRENGTH BENDER STATS", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 75, 173), 2)
                        y = 130
                        for r in rep_data:
                            cv2.putText(card, f"R{r['id']}: {r['duration']:.2f}s | {r['avg_v']:.2f}m/s", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(card, f"PEAK: {r['peak_v']:.2f}m/s", (60, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            y += 70
                        for _ in range(int(fps * 4)): out_frames.append(card)

                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    imageio.mimsave(final_path, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(final_path)
