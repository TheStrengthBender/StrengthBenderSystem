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

# Fix for the "Infinite Click" loop
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

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
        h, w = first_frame.shape[:2]
        new_width = 400
        new_height = int(new_width * (h / w))
        if new_height % 2 != 0: new_height += 1
        
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_resized = cv2.resize(frame_rgb, (new_width, new_height))

        # Only show the clicker if we haven't successfully processed a click yet
        if not st.session_state.clicked:
            st.markdown("### 🎯 Step 1: Click on the Barbell")
            value = streamlit_image_coordinates(first_frame_resized, key="clicker_stable")
            if value:
                st.session_state.coords = (value['x'], value['y'])
                st.session_state.clicked = True
                st.rerun()

        if st.session_state.clicked:
            cx, cy = st.session_state.coords
            first_frame_bgr = cv2.resize(first_frame, (new_width, new_height))
            bbox = (cx - 25, cy - 25, 50, 50)
            
            st.info("Target Locked. Analyzing Set...")

            tracker = cv2.TrackerCSRT_create() 
            tracker.init(first_frame_bgr, bbox)

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

            # --- ROBUST HYBRID MATH (v8.0) ---
            smoothed_y = [np.mean(y_history[max(0, i-5):min(len(y_history), i+5)]) for i in range(len(y_history))]
            meters_per_pixel = 0.45 / bboxes[0][3]
            
            # Identify all 'lowest' points
            rep_starts = []
            for i in range(15, len(smoothed_y) - 15):
                if smoothed_y[i] == max(smoothed_y[i-15:i+16]):
                    # Valid bottom if it's deep enough from the 'average' height
                    if (smoothed_y[i] - min(smoothed_y)) > 25:
                        if not rep_starts or (i - rep_starts[-1]) > (fps * 1.0):
                            rep_starts.append(i)

            rep_data = []
            for start in rep_starts:
                search_range = smoothed_y[start:min(start + int(fps * 3), len(smoothed_y))]
                end = start
                for j in range(1, len(search_range)):
                    # Precise Lockout Snap
                    if search_range[j] >= search_range[j-1]:
                        end = start + j
                        break
                
                # Check displacement (Did it actually move?)
                dist = (smoothed_y[start] - smoothed_y[end]) * meters_per_pixel
                if dist > 0.15: # 15cm minimum for Trap Bar pulls
                    rep_y = y_history[start:end+1]
                    v_raw = [abs(rep_y[k-1] - rep_y[k]) * meters_per_pixel * fps for k in range(1, len(rep_y))]
                    rep_data.append({
                        "id": len(rep_data) + 1, "start": start, "end": end,
                        "avg_v": np.mean(v_raw), "duration": (end - start) / fps
                    })

            # --- FINAL BAKE ---
            col1, col2 = st.columns([3, 1])
            with col2:
                st.subheader("📊 System Stats")
                for r in rep_data:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} m/s | {r["duration"]:.2f}s</div>', unsafe_allow_html=True)
                if st.button("Reset System"):
                    st.session_state.clicked = False
                    st.rerun()

            with col1:
                with st.spinner("Baking Final Clip..."):
                    out_frames, path_points = [], []
                    for i in range(len(all_frames)):
                        frame_draw = all_frames[i].copy()
                        bx, by, bw, bh = bboxes[i]
                        active_rep = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                        path_points.append((bx + bw//2, by + bh//2))
                        
                        # Draw Path
                        if len(path_points) > 1:
                            for j in range(max(1, i-60), len(path_points)):
                                cv2.line(frame_draw, path_points[j-1], path_points[j], (255, 75, 173), 2, cv2.LINE_AA)

                        # HUD
                        cv2.rectangle(frame_draw, (0, 0), (new_width, 60), (0, 0, 0), -1)
                        if active_rep:
                            cv2.putText(frame_draw, f"REP {active_rep['id']} | {(i-active_rep['start'])/fps:.2f}s", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        else:
                            cv2.putText(frame_draw, "STRENGTH BENDER READY", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        out_frames.append(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB))

                    # Final Summary Frame
                    if rep_data:
                        summary_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                        cv2.putText(summary_frame, "THE STRENGTH BENDER", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 75, 173), 2)
                        y_off = 120
                        for r in rep_data:
                            cv2.putText(summary_frame, f"R{r['id']}: {r['duration']:.2f}s | {r['avg_v']:.2f}m/s", (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            y_off += 40
                        for _ in range(int(fps * 3)): out_frames.append(summary_frame)

                    imageio.mimsave(tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name, out_frames, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name)
