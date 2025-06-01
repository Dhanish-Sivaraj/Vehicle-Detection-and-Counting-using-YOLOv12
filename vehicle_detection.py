import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
import time

# Page Config
st.set_page_config(
    page_title="🚗 YOLOv12 Vehicle Counter",
    layout="wide"
)

# Session State
if 'counts' not in st.session_state:
    st.session_state.counts = defaultdict(int)
if 'tracked' not in st.session_state:
    st.session_state.tracked = {}
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None

# --- SIMPLIFIED MODEL LOADING ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_files = [
        'yolov12n.pt',  # Primary model to try
        'yolov12.pt',   # Alternate naming
        'yolov8n.pt'    # Fallback
    ]
    
    for model_file in model_files:
        try:
            full_path = os.path.join(current_dir, model_file)
            if os.path.exists(full_path):
                model = YOLO(full_path)
                st.success(f"✅ Model loaded: {os.path.basename(full_path)}")
                return model
        except Exception:
            continue
    
    st.error("❌ Could not load any YOLO model!")
    return None

model = load_model()
if model is None:
    st.stop()

# Vehicle Classes (Update these based on your YOLOv12's class IDs)
VEHICLE_CLASSES = {
    2: 'car', 
    3: 'motorcycle', 
    5: 'bus', 
    7: 'truck',
    1: 'bicycle',
    6: 'train'
}

# UI Settings
st.sidebar.title("⚙️ Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4)
tracking_enabled = st.sidebar.checkbox("Enable Tracking", True)
min_frames_to_count = st.sidebar.slider("Min Frames to Count", 5, 30, 10)
downsample = st.sidebar.slider("Process Every N Frames", 1, 5, 2)

# Display current model info
st.sidebar.markdown("---")
st.sidebar.write(f"**Current Model:** {model.__class__.__name__}")
st.sidebar.write(f"**Model Path:** {model.ckpt_path}")

# Main App
st.title(f"🚦 Vehicle Detection & Counting (YOLOv12)")
st.write("**Counts each vehicle only once** using tracking")

# Tracking Logic
class VehicleTracker:
    def __init__(self):
        self.tracked = {}
        self.counted = set()
        self.next_id = 1

    def _iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = (x2-x1)*(y2-y1) + (x2_p-x1_p)*(y2_p-y1_p) - inter
        
        return inter / union if union > 0 else 0

    def update(self, detections, frame_num):
        new_counts = defaultdict(int)
        matched = set()
        updated = {}
        
        for track_id, vehicle in self.tracked.items():
            best_iou = 0
            best_match = None
            
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                
                iou = self._iou(vehicle['box'], det['box'])
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                matched.add(best_match)
                det = detections[best_match]
                
                updated[track_id] = {
                    'box': det['box'],
                    'class': det['class'],
                    'first_frame': vehicle['first_frame'],
                    'last_frame': frame_num,
                    'counted': vehicle['counted']
                }
                
                if not vehicle['counted'] and (frame_num - vehicle['first_frame']) >= min_frames_to_count:
                    updated[track_id]['counted'] = True
                    new_counts[det['class']] += 1
                    self.counted.add(track_id)
        
        for i, det in enumerate(detections):
            if i not in matched:
                track_id = f"V{self.next_id}"
                self.next_id += 1
                updated[track_id] = {
                    'box': det['box'],
                    'class': det['class'],
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'counted': False
                }
        
        self.tracked = updated
        return new_counts

# Video Processing
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = os.path.join(tempfile.gettempdir(), "output_counted.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    tracker = VehicleTracker()
    frame_num = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    video_placeholder = st.empty()
    counts_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        if frame_num % downsample != 0:
            continue
        
        # Detect vehicles with YOLO
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': VEHICLE_CLASSES[cls_id],
                        'conf': box.conf.item()
                    })
        
        new_counts = tracker.update(detections, frame_num)
        
        for cls, count in new_counts.items():
            st.session_state.counts[cls] += count
        
        for track_id, vehicle in tracker.tracked.items():
            x1, y1, x2, y2 = vehicle['box']
            color = (0, 255, 0) if vehicle['counted'] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{vehicle['class']} {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        
        progress = frame_num / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar.progress(progress)
        
        if frame_num % 5 == 0:
            video_placeholder.image(frame, channels="BGR", use_column_width=True)
            counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient='index', columns=['Count'])
            counts_placeholder.dataframe(counts_df)
        
        status_text.text(f"Processed Frame {frame_num} | Total Counts: {sum(st.session_state.counts.values())}")
    
    cap.release()
    out.release()
    st.session_state.processed_video = output_path
    return output_path

# File Upload
uploaded_file = st.file_uploader("📤 Upload Traffic Video", type=["mp4", "mov", "avi"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    
    st.video(video_path)
    
    if st.button("🚀 Start Counting"):
        st.session_state.counts = defaultdict(int)
        st.session_state.tracked = {}
        
        with st.spinner("Processing video..."):
            output_path = process_video(video_path)
            st.success("✅ Processing complete!")
            st.video(output_path)
            
            st.subheader("📊 Final Counts")
            counts_df = pd.DataFrame.from_dict(st.session_state.counts, orient='index', columns=['Count'])
            st.dataframe(counts_df)
            
            st.download_button(
                "📥 Download Results",
                counts_df.to_csv().encode('utf-8'),
                "vehicle_counts.csv"
            )

# Debug information
st.markdown("---")
st.subheader("🔍 Debug Information")
st.write(f"Current working directory: `{os.getcwd()}`")
st.write(f"Files in directory: `{os.listdir()}`")

st.markdown("---")
st.write("**How it works:**")
st.write("1. Detects vehicles frame-by-frame using YOLOv12")
st.write("2. Tracks each vehicle with a unique ID")
st.write("3. Counts only after seeing a vehicle for N frames")
st.write("4. **Green** = counted, **Red** = tracking")