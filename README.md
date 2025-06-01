# 🚗 Vehicle Detection and Counting using YOLOv12

This Streamlit web app detects and counts vehicles (car, bike, bus, etc.) in a video using the YOLOv12 object detection model. It tracks each vehicle to ensure accurate counting.

## 📂 Project Structure

```
├── vehicle_detection.py          # Main Streamlit App
├── yolov12n.pt                   # YOLOv12 model weights
├── requirements.txt              # Required Python packages
├── sample_video.mp4              # (Optional) Sample input video
└── README.md                     # Project overview
```

## 🔧 Features

- Vehicle detection using YOLOv12
- Real-time vehicle counting
- Object tracking to avoid duplicate counts
- Adjustable confidence and frame settings
- Downloadable count results (CSV)

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vehicle-detection-yolov12.git
cd vehicle-detection-yolov12
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run vehicle_detection.py
```

### 4. Upload a traffic video and start counting!

## 📦 Dependencies

- streamlit
- opencv-python
- numpy
- pandas
- ultralytics
- torch

## 📸 Demo

![image](https://github.com/user-attachments/assets/67df2063-699b-4395-9676-53ae6c4d6923)


## 🧠 Model Info

YOLOv12n custom model trained for vehicle detection (supports cars, bikes, trucks, buses, etc.)


## 🙋‍♂️ Author

Dhanish S
