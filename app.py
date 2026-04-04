import streamlit as st
import cv2
import imageio
import numpy as np
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="TheStrengthBenderSystem", page_icon="🏋️", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .stButton > button { background-color: #FF4BAD !important; color: white !important; font-weight: bold; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏋️ StrengthBender Pro")

uploaded_file = st.file_uploader("Upload Set (MP4)", type=["mp4"])

if uploaded_file is not None:
    if st.button("🚀 ANALYZE LIFT"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        
        if ret:
            h, w = first_frame.shape[:2]
            new_width = 400
            new_height = int(new_width * (h / w))
            if new_height % 2 != 0: new_height += 1
            first_frame_res = cv2.resize(first_frame, (new_width, new_height))

            st.write("### 🎯 Click the Barbell")
            value = streamlit_image_coordinates(cv2.cvtColor(first_frame_res, cv2.COLOR_BGR2RGB), key="click")

            if value:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(first_frame_res, (value['x']-25, value['y']-25, 50, 50))
                
                y_hist, bboxes, frames = [], [], []
                prog = st.progress(0)
                
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    f_res = cv2.resize(frame, (new_width, new_height))
                    ok, box = tracker.update(f_res)
                    res_box = [int(v) for v in box] if ok else (bboxes[-1] if bboxes else [0,0,0,0])
                    y_hist.append(res_box[1] + res_box[3]//2)
                    bboxes.append(res_box)
                    frames.append(f_res)
                    prog.progress((i+1)/total_frames)

                # --- FAST REP DETECTION ---
                meters_px = 0.45 / bboxes[0][3]
                smoothed_y = [np.mean(y_hist[max(0, x-5):min(len(y_hist), x+5)]) for x in range(len(y_hist))]
                reps = []
                for i in range(15, len(smoothed_y)-15):
                    if smoothed_y[i] == max(smoothed_y[i-15:i+16]) and (smoothed_y[i] - min(smoothed_y)) > 30:
                        if not reps or (i - reps[-1]['start']) > fps:
                            # Simple search for lockout
                            end = i
                            for j in range(i+5, min(i+int(fps*3), len(smoothed_y))):
                                if smoothed_y[j] >= smoothed_y[j-1]: 
                                    end = j; break
                            
                            v_raw = [abs(y_hist[k-1]-y_hist[k])*meters_px*fps for k in range(i, end+1)]
                            if v_raw and (max(y_hist[i:end+1]) - min(y_hist[i:end+1]))*meters_px > 0.1:
                                reps.append({"id": len(reps)+1, "start": i, "end": end, "avg": np.mean(v_raw)})

                # --- BURN STATS INTO VIDEO ---
                with st.spinner("Baking Video + Stats..."):
                    baked = []
                    path = []
                    for i in range(len(frames)):
                        img = frames[i].copy()
                        path.append((bboxes[i][0]+bboxes[i][2]//2, bboxes[i][1]+bboxes[i][3]//2))
                        
                        # Draw Path
                        for j in range(max(1, i-50), len(path)):
                            cv2.line(img, path[j-1], path[j], (255, 0, 255), 2)
                        
                        # HUD
                        cv2.rectangle(img, (0,0), (new_width, 60), (0,0,0), -1)
                        curr = next((r for r in reps if r['start'] <= i <= r['end']), None)
                        if curr:
                            cv2.putText(img, f"REP {curr['id']} | {(i-curr['start'])/fps:.2f}s", (10, 25), 0, 0.6, (0,255,255), 2)
                            cv2.putText(img, f"VEL: {curr['avg']:.2f} m/s", (10, 50), 0, 0.5, (255,255,255), 1)
                        else:
                            cv2.putText(img, "STRENGTH BENDER SYSTEM", (10, 35), 0, 0.5, (150,150,150), 1)
                        
                        baked.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    # FINAL SUMMARY FRAME
                    sum_f = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    cv2.putText(sum_f, "SYSTEM STATS", (100, 50), 0, 0.7, (255, 0, 255), 2)
                    for idx, r in enumerate(reps):
                        cv2.putText(sum_f, f"R{r['id']}: {r['avg']:.2f}m/s | {(r['end']-r['start'])/fps:.2f}s", (50, 100+(idx*40)), 0, 0.5, (255,255,255), 1)
                    for _ in range(int(fps*3)): baked.append(sum_f)

                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    imageio.mimsave(out_path, baked, fps=fps, format='FFMPEG', codec='libx264')
                    st.video(out_path)
