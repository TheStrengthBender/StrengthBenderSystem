import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1 { color: #E0E0E0 !important; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .rep-card { background-color: #1A1C23; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4BAD; margin-bottom: 10px; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ TheStrengthBenderSystem")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

uploaded_file = st.file_uploader("Upload Set (MP4 or MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
    with open(tpath, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(tpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if ret:
        h, w = first_frame.shape[:2]
        new_w, new_h = 400, int(400 * (h / w))
        if new_h % 2 != 0: new_h += 1
        
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_res = cv2.resize(frame_rgb, (new_w, new_h))

        if not st.session_state.clicked:
            st.markdown("### 🎯 Step 1: Click on the Barbell")
            value = streamlit_image_coordinates(first_frame_res, key="clicker")
            if value:
                st.session_state.coords = (value['x'], value['y'])
                st.session_state.clicked = True
                st.rerun()

        if st.session_state.clicked:
            cx, cy = st.session_state.coords
            tracker = cv2.TrackerCSRT_create()
            first_frame_bgr = cv2.resize(first_frame, (new_w, new_h))
            tracker.init(first_frame_bgr, (cx-25, cy-25, 50, 50))

            y_hist, bboxes, frames = [], [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progress = st.progress(0)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                f_res = cv2.resize(frame, (new_w, new_h))
                ok, box = tracker.update(f_res)
                bx, by, bw, bh = [int(v) for v in (box if ok else bboxes[-1] if bboxes else (cx-25, cy-25, 50, 50))]
                y_hist.append(by + bh//2)
                bboxes.append((bx, by, bw, bh))
                frames.append(cv2.cvtColor(f_res, cv2.COLOR_BGR2RGB))
                progress.progress((i + 1) / total_frames)

            # --- THE "ANY-PULL" ENGINE ---
            # Smoothed velocity instead of just position
            m_per_px = 0.45 / bboxes[0][3]
            v_instant = [0]
            for i in range(1, len(y_hist)):
                v_instant.append((y_hist[i-1] - y_hist[i]) * m_per_px * fps)
            
            v_smooth = [np.mean(v_instant[max(0, x-3):min(len(v_instant), x+3)]) for x in range(len(v_instant))]
            
            rep_data = []
            is_moving = False
            start_f = 0
            
            for i, v in enumerate(v_smooth):
                # Start tracking a rep if velocity > 0.1 m/s
                if not is_moving and v > 0.1:
                    is_moving = True
                    start_f = i
                # Stop tracking if velocity hits 0 or negative
                elif is_moving and v <= 0:
                    end_f = i
                    duration = (end_f - start_f) / fps
                    # Only count if rep is > 0.3s and moves > 10cm
                    dist = (y_hist[start_f] - y_hist[end_f]) * m_per_px
                    if duration > 0.3 and dist > 0.10:
                        rep_v = v_smooth[start_f:end_f+1]
                        rep_data.append({
                            "id": len(rep_data)+1, 
                            "start": start_f, "end": end_f, 
                            "avg_v": np.mean(rep_v), 
                            "dur": duration
                        })
                    is_moving = False

            # --- DISPLAY ---
            c1, c2 = st.columns([3, 1])
            with c2:
                st.subheader("📊 System Stats")
                for r in rep_data:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} m/s | {r["dur"]:.2f}s</div>', unsafe_allow_html=True)
                if st.button("Reset"):
                    st.session_state.clicked = False
                    st.rerun()

            with c1:
                path_pts, out_frames = [], []
                for i, f in enumerate(frames):
                    bx, by, bw, bh = bboxes[i]
                    path_pts.append((bx+bw//2, by+bh//2))
                    active = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                    
                    if len(path_pts) > 1:
                        for j in range(max(1, i-60), len(path_pts)):
                            cv2.line(f, path_pts[j-1], path_pts[j], (255, 75, 173), 2)
                    
                    cv2.rectangle(f, (0,0), (new_w, 50), (0,0,0), -1)
                    txt = f"REP {active['id']} | {(i-active['start'])/fps:.2f}s" if active else "STRENGTH BENDER READY"
                    cv2.putText(f, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    out_frames.append(f)

                # End Card
                if rep_data:
                    card = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    cv2.putText(card, "SYSTEM STATS", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 75, 173), 2)
                    for idx, r in enumerate(rep_data):
                        cv2.putText(card, f"R{r['id']}: {r['dur']:.2f}s | {r['avg_v']:.2f}m/s", (40, 120 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    for _ in range(int(fps*3)): out_frames.append(card)

                final_p = os.path.join(tempfile.gettempdir(), "out.mp4")
                imageio.mimsave(final_p, out_frames, fps=fps, codec='libx264')
                st.video(final_p)
