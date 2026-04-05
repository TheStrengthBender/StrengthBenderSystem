import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
import os
from streamlit_image_coordinates import streamlit_image_coordinates

# --- PRO UI CONFIG ---
st.set_page_config(page_title="The Strength Bender", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Sleek typography */
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 800; tracking: tight; }
    
    /* Premium Rep Cards */
    .rep-card { 
        background: linear-gradient(145deg, #1A1C23 0%, #12141A 100%);
        padding: 20px; 
        border-radius: 12px; 
        border-left: 4px solid #FF4BAD; 
        margin-bottom: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .rep-card:hover { transform: translateY(-2px); }
    
    .rep-header { color: #FF4BAD; font-size: 1.1rem; font-weight: 800; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .stat-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-family: 'SF Pro Display', monospace; }
    .stat-label { color: #A0AEC0; font-size: 0.9rem; font-weight: 600; }
    .stat-value { color: #FFFFFF; font-size: 1.1rem; font-weight: 700; }
    .peak-val { color: #00E6FF; } /* Cyan for peak */
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# State Management
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# --- SIDEBAR: COMMAND CENTER ---
with st.sidebar:
    st.title("⚡ Settings")
    st.markdown("Upload your lift to begin analysis.")
    uploaded_file = st.file_uploader("Upload Set (MP4/MOV)", type=["mp4", "mov"], label_visibility="collapsed")
    
    if st.session_state.clicked:
        st.markdown("---")
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.clicked = False
            st.rerun()

# --- MAIN STAGE ---
st.title("TheStrengthBender System")

if uploaded_file is not None:
    tpath = os.path.join(tempfile.gettempdir(), "input.mp4")
    with open(tpath, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(tpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if ret:
        orig_h, orig_w = first_frame.shape[:2]
        display_w = 400
        scale_factor = orig_w / display_w
        
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame_res = cv2.resize(frame_rgb, (display_w, int(display_w * (orig_h / orig_w))))

        # --- STEP 1: INITIALIZATION ---
        if not st.session_state.clicked:
            st.markdown("### 🎯 **Target Acquisition:** Click the tip of the sleeve or the bar")
            value = streamlit_image_coordinates(first_frame_res, key="clicker_v11")
            if value:
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

        # --- STEP 2: PHYSICS ENGINE ---
        if st.session_state.clicked:
            cx, cy = st.session_state.coords
            tracker = cv2.TrackerCSRT_create()
            
            orig_cx, orig_cy = int(cx * scale_factor), int(cy * scale_factor)
            box_size = int(50 * scale_factor)
            tracker.init(first_frame, (orig_cx - box_size//2, orig_cy - box_size//2, box_size, box_size))

            y_hist_orig, bboxes_orig, frames_display = [], [], []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            with st.spinner("Processing Lift Kinematics..."):
                progress = st.progress(0)
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    ok, box = tracker.update(frame)
                    
                    bx, by, bw, bh = [int(v) for v in (box if ok else bboxes_orig[-1] if bboxes_orig else (orig_cx-box_size//2, orig_cy-box_size//2, box_size, box_size))]
                    y_hist_orig.append(by + bh//2)
                    bboxes_orig.append((bx, by, bw, bh))
                    
                    f_disp = cv2.resize(frame, (display_w, int(display_w * (orig_h / orig_w))))
                    frames_display.append(cv2.cvtColor(f_disp, cv2.COLOR_BGR2RGB))
                    progress.progress((i + 1) / total_frames)
                progress.empty() # Clear progress bar when done

            # --- MATH ENGINE (PEAK VELOCITY RESTORED) ---
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
                        rep_v_array = v_smooth[start_f:end_f+1]
                        rep_data.append({
                            "id": len(rep_data)+1, 
                            "start": start_f, "end": end_f, 
                            "avg_v": np.mean(rep_v_array), 
                            "peak_v": max(rep_v_array), # BACK IN ACTION
                            "dur": (end_f - start_f) / fps
                        })
                    is_moving = False

            # --- STEP 3: DASHBOARD DISPLAY ---
            # Phone: Stacks vertically. PC: Side-by-side (2/3 Video, 1/3 Stats)
            col_vid, col_stats = st.columns([2, 1.2], gap="large")
            
            with col_vid:
                path_pts_disp, out_frames = [], []
                for i, f in enumerate(frames_display):
                    bx, by, bw, bh = [int(v / scale_factor) for v in bboxes_orig[i]]
                    path_pts_disp.append((bx+bw//2, by+bh//2))
                    active = next((r for r in rep_data if r['start'] <= i <= r['end']), None)
                    
                    if len(path_pts_disp) > 1:
                        draw_start = active['start'] if active else max(1, i - 30)
                        for j in range(max(1, draw_start), len(path_pts_disp)):
                            if path_pts_disp[j][1] > path_pts_disp[j-1][1]:
                                path_color = (255, 255, 0) # Cyan (Eccentric)
                            else:
                                path_color = (255, 75, 173) # Pink (Concentric)
                            cv2.line(f, path_pts_disp[j-1], path_pts_disp[j], path_color, 2, cv2.LINE_AA)
                    
                    cv2.rectangle(f, (0,0), (display_w, 50), (0,0,0), -1)
                    txt = f"REP {active['id']} | {(i-active['start'])/fps:.2f}s" if active else "READY"
                    cv2.putText(f, txt, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    out_frames.append(f)

                final_p = os.path.join(tempfile.gettempdir(), "out.mp4")
                imageio.mimsave(final_p, out_frames, fps=fps, codec='libx264')
                st.video(final_p)

            with col_stats:
                st.markdown("### Session Stats")
                if not rep_data:
                    st.info("No valid reps detected. Ensure bar moved > 30cm.")
                
                for r in rep_data:
                    # Beautiful HTML Cards replacing the old text
                    st.markdown(f"""
                        <div class="rep-card">
                            <div class="rep-header">REP {r["id"]}</div>
                            <div class="stat-row">
                                <span class="stat-label">AVG VELOCITY</span>
                                <span class="stat-value">{r["avg_v"]:.2f} m/s</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">PEAK VELOCITY</span>
                                <span class="stat-value peak-val">{r["peak_v"]:.2f} m/s</span>
                            </div>
                            <div class="stat-row" style="margin-top: 8px; border-top: 1px solid #2D3748; padding-top: 8px;">
                                <span class="stat-label">DURATION</span>
                                <span class="stat-value">{r["dur"]:.2f}s</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
else:
    # Welcome Screen when no video is uploaded
    st.info("👈 Open the sidebar menu to upload a video and begin.")
