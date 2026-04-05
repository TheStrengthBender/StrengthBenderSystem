import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
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
        # --- CALIBRATION CONSTANTS ---
        orig_h, orig_w = first_frame.shape[:2]
        display_w = 400
        scale_factor = orig_w / display_w
        
        # Display frame for clicking
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_res = cv2.resize(frame_rgb, (display_w, int(display_w * (orig_h / orig_w))))

        if not st.session_state.clicked:
            st.markdown("### 🎯 Step 1: Click the tip of the sleeve or the bar")
            value = streamlit_image_coordinates(first_frame_res, key="clicker_v10")
            if value:
                # Snap to edge
                gray = cv2.cvtColor(first_frame_res, cv2.COLOR_RGB2GRAY)
                y1, y2 = max(0, value['y']-10), min(first_frame_res.shape[0], value['y']+10)
                x1, x2 = max(0, value['x']-10), min(display_w, value['x']+10)
                roi = gray[y1:y2, x1:x2]
                grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                mag = cv2.magnitude(grad_x, grad_y)
                _, _, _, max_loc = cv2.minMaxLoc(mag)
                
                st.session_state.coords = (x1 + max_loc[0], y1 + max_loc[1])
                st.session_state.clicked = True
                st.rerun()

        if st.session_state.clicked:
            cx, cy = st.session_state.coords
            tracker = cv2.TrackerCSRT_create()
            
            # TRACK IN ORIGINAL RESOLUTION FOR MAX ACCURACY
            orig_cx, orig_cy = int(cx * scale_factor), int(cy * scale_factor)
            box_size = int(50 * scale_factor)
            tracker.init(first_frame, (orig_cx - box_size//2, orig_cy - box_size//2, box_size, box_size))

            y_hist_orig, bboxes_orig, frames_display = [], [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progress = st.progress(0)
            
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                ok, box = tracker.update(frame)
                
                # Math stays in high-res
                bx, by, bw, bh = [int(v) for v in (box if ok else bboxes_orig[-1] if bboxes_orig else (orig_cx-box_size//2, orig_cy-box_size//2, box_size, box_size))]
                y_hist_orig.append(by + bh//2)
                bboxes_orig.append((bx, by, bw, bh))
                
                # Display is resized
                f_disp = cv2.resize(frame, (display_w, int(display_w * (orig_h / orig_w))))
                frames_display.append(cv2.cvtColor(f_disp, cv2.COLOR_BGR2RGB))
                progress.progress((i + 1) / total_frames)

            # --- HIGH-RES PHYSICS ENGINE ---
            # Standard Olympic sleeve diameter is 50mm, but plates are 450mm. 
            # We use the initial box height (bh) as our 0.45m reference.
            m_per_px = 0.45 / bboxes_orig[0][3]
            v_instant = [0]
            for i in range(1, len(y_hist_orig)):
                v_instant.append((y_hist_orig[i-1] - y_hist_orig[i]) * m_per_px * fps)
            
            v_smooth = [np.mean(v_instant[max(0, x-3):min(len(v_instant), x+3)]) for x in range(len(v_instant))]
            
            rep_data = []
            is_moving = False
            start_f = 0
            
            for i, v in enumerate(v_smooth):
                if not is_moving and v > 0.15:
                    is_moving = True
                    start_f = i
                elif is_moving and v <= 0:
                    end_f = i
                    dist = (y_hist_orig[start_f] - y_hist_orig[end_f]) * m_per_px
                    if dist > 0.30:
                        rep_data.append({
                            "id": len(rep_data)+1, 
                            "start": start_f, "end": end_f, 
                            "avg_v": np.mean(v_smooth[start_f:end_f+1]), 
                            "dur": (end_f - start_f) / fps
                        })
                    is_moving = False

            # --- DISPLAY ---
            c1, c2 = st.columns([3, 1])
            with c2:
                st.subheader("📊 System Stats")
                for r in rep_data:
                    st.markdown(f'<div class="rep-card"><b>REP {r["id"]}</b><br>{r["avg_v"]:.2f} m/s | {r["dur"]:.2f}s</div>', unsafe_allow_html=True)
                if st.button("Reset System"):
                    st.session_state.clicked = False
                    st.rerun()

           with c1:
                path_pts_disp, out_frames = [], []
                for i, f in enumerate(frames_display):
                    bx, by, bw, bh = [int(v / scale_factor) for v in bboxes_orig[i]]
                    path_pts_disp.append((bx+bw//2, by+bh//2))
                    active = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                    
                    if len(path_pts_disp) > 1:
                        # --- PERSISTENT DUAL-COLOR PATH ---
                        # Anchor the line to the start of the rep, or use a short tail if just resting
                        draw_start = active['start'] if active else max(1, i - 30)
                        
                        for j in range(max(1, draw_start), len(path_pts_disp)):
                            # OpenCV Y-axis goes DOWN. If current Y > prev Y, bar is descending.
                            if path_pts_disp[j][1] > path_pts_disp[j-1][1]:
                                path_color = (255, 255, 0) # Cyan for Descent (Eccentric)
                            else:
                                path_color = (255, 75, 173) # Pink for Ascent (Concentric)
                                
                            # Added LINE_AA for a smoother, less pixelated path
                            cv2.line(f, path_pts_disp[j-1], path_pts_disp[j], path_color, 2, cv2.LINE_AA)
                    
                    cv2.rectangle(f, (0,0), (display_w, 50), (0,0,0), -1)
                    txt = f"REP {active['id']} | {(i-active['start'])/fps:.2f}s" if active else "STRENGTH BENDER READY"
                    cv2.putText(f, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    out_frames.append(f)

                if rep_data:
                    card = np.zeros((frames_display[0].shape[0], display_w, 3), dtype=np.uint8)
                    cv2.putText(card, "SYSTEM STATS", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 75, 173), 2)
                    for idx, r in enumerate(rep_data):
                        cv2.putText(card, f"R{r['id']}: {r['dur']:.2f}s | {r['avg_v']:.2f}m/s", (40, 120 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    for _ in range(int(fps*3)): out_frames.append(card)

                final_p = os.path.join(tempfile.gettempdir(), "out.mp4")
                imageio.mimsave(final_p, out_frames, fps=fps, codec='libx264')
                st.video(final_p)
