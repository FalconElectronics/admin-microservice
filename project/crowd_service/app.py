# crowd_service/app.py
from fastapi import FastAPI, HTTPException
import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
import torch
from ultralytics import YOLO
import asyncio
import redis
import sys
from fastapi.middleware.cors import CORSMiddleware

# Add ByteTrack to path - modify as needed for your environment
sys.path.append(os.getenv("BYTETRACK_PATH", "ByteTrack"))
from yolox.tracker.byte_tracker import BYTETracker

app = FastAPI(title="Crowd Analytics Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for storing results and inter-service communication
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Load crowd detection model
model = YOLO(os.getenv("CROWD_MODEL_PATH", "best_yolo11s_crowd.engine"))

# Setup for tracking
class ByteTrackArgs:
    def __init__(self, track_thresh=0.25, match_thresh=0.5, track_buffer=80, mot20=False):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.mot20 = mot20
        self.min_box_area = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
tracking_data = {}
camera_stats = {
    "cam1": {"entries": 0, "exits": 0, "people_count": 0},
    "cam2": {"entries": 0, "exits": 0, "people_count": 0},
    "cam3": {"entries": 0, "exits": 0, "people_count": 0}
}
crossed_line = {}

def check_line_crossing(track_id, trajectory, line_y, cam_id):
    """Detect if a person crosses a virtual line"""
    if len(trajectory) < 2:
        return
    
    prev_x1, prev_y1, prev_x2, prev_y2 = trajectory[-2]
    curr_x1, curr_y1, curr_x2, curr_y2 = trajectory[-1]
    prev_center_y = (prev_y1 + prev_y2) // 2
    curr_center_y = (curr_y1 + curr_y2) // 2

    # Detect entry
    if prev_center_y < line_y and curr_center_y >= line_y:
        if track_id not in crossed_line or crossed_line[track_id] != "entered":
            crossed_line[track_id] = "entered"
            camera_stats[cam_id]["entries"] += 1
    
    # Detect exit
    elif prev_center_y > line_y and curr_center_y <= line_y:
        if track_id not in crossed_line or crossed_line[track_id] != "exited":
            crossed_line[track_id] = "exited"
            camera_stats[cam_id]["exits"] += 1

def calculate_movement(trajectory):
    """Calculate speed and direction of movement"""
    if len(trajectory) < 2:
        return 0, "Stationary"
    
    prev = trajectory[-2]
    curr = trajectory[-1]
    prev_center = ((prev[0] + prev[2]) // 2, (prev[1] + prev[3]) // 2)
    curr_center = ((curr[0] + curr[2]) // 2, (curr[1] + curr[3]) // 2)
    
    speed = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    
    if abs(dx) > abs(dy):
        direction = "Right" if dx > 0 else "Left"
    else:
        direction = "Down" if dy > 0 else "Up"
    
    return speed, direction

def detect_anomaly(speed_history):
    """Detect anomalies in movement patterns"""
    if len(speed_history) < 5:
        return "Normal"
    
    avg_speed = np.mean(speed_history[-5:])
    current_speed = speed_history[-1]
    
    if current_speed > avg_speed * 2:
        return "Anomaly: Sudden Speed Change"
    if current_speed < avg_speed * 0.5 and current_speed > 0:
        return "Anomaly: Sudden Stop"
    
    return "Normal"

async def process_frame(frame, cam_id, tracker):
    """Process a frame for crowd analytics"""
    results = model(frame)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if conf >= 0.8:
                detections.append([x1, y1, x2, y2, conf])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    dets = np.array(detections) if detections else np.empty((0, 5))
    tracks = tracker.update(dets, (frame.shape[0], frame.shape[1]), (frame.shape[0], frame.shape[1]))
    
    camera_stats[cam_id]["people_count"] = len(tracks)
    
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.tlbr)
        track_id = track.track_id
        
        if track_id not in tracking_data:
            tracking_data[track_id] = {
                "camera": cam_id,
                "trajectory": [],
                "start_time": time.time(),
                "speed_history": [],
                "current_speed": 0,
                "movement_direction": "Stationary",
                "anomaly": "Normal",
                "dwell_time": 0
            }
        
        tracking_data[track_id]["trajectory"].append([x1, y1, x2, y2])
        tracking_data[track_id]["dwell_time"] = time.time() - tracking_data[track_id]["start_time"]
        
        speed, direction = calculate_movement(tracking_data[track_id]["trajectory"])
        tracking_data[track_id]["speed_history"].append(speed)
        tracking_data[track_id]["current_speed"] = speed
        tracking_data[track_id]["movement_direction"] = direction
        tracking_data[track_id]["anomaly"] = detect_anomaly(tracking_data[track_id]["speed_history"])
        
        # Define virtual line (use 200 for cam1, 300 for others)
        line_y = 200 if cam_id == "cam1" else 300
        check_line_crossing(track_id, tracking_data[track_id]["trajectory"], line_y, cam_id)
        
        # Draw bounding box with speed and direction info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id} ({direction}, {speed:.2f}px/frame)", (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

async def continuous_crowd_analytics(camera_sources):
    """Continuously process frames from multiple camera sources for crowd analytics"""
    trackers = {
        cam_id: BYTETracker(ByteTrackArgs()) for cam_id in camera_sources
    }
    
    caps = {}
    
    # Initialize video captures
    for cam_id, source in camera_sources.items():
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {cam_id}")
            continue
        caps[cam_id] = cap
    
    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Camera {cam_id} feed lost.")
                continue
            
            processed_frame = await process_frame(frame, cam_id, trackers[cam_id])
            
            # Store the processed frame for streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            redis_client.set(f"crowd_frame:{cam_id}", buffer.tobytes())
            
            # Update Redis with latest tracking data and stats
            redis_client.hset("crowd_tracking", cam_id, json.dumps({
                "stats": camera_stats[cam_id],
                "tracks": {k: v for k, v in tracking_data.items() if v["camera"] == cam_id}
            }))
        
        await asyncio.sleep(0.03)  # Throttle processing

@app.on_event("startup")
async def startup_event():
    # Define camera sources - in production, load from config
    camera_sources = {
        "cam1": os.getenv("CAMERA1_URL", "rtsp://192.168.1.121:554/stream0"),
        "cam2": os.getenv("CAMERA2_URL", "rtsp://192.168.1.30:554/stream0"),
        "cam3": os.getenv("CAMERA3_URL", "rtsp://192.168.1.27:554/stream0")
    }
    
    # Start continuous analytics in background
    asyncio.create_task(continuous_crowd_analytics(camera_sources))

@app.get("/analytics")
async def get_analytics():
    """Get all crowd analytics data"""
    return {
        "tracking_data": tracking_data,
        "camera_stats": camera_stats
    }

@app.get("/analytics/{cam_id}")
async def get_camera_analytics(cam_id: str):
    """Get analytics for a specific camera"""
    camera_tracking = {k: v for k, v in tracking_data.items() if v["camera"] == cam_id}
    return {
        "tracking_data": camera_tracking,
        "stats": camera_stats.get(cam_id, {"entries": 0, "exits": 0, "people_count": 0})
    }

@app.get("/frame/{cam_id}")
async def get_latest_frame(cam_id: str):
    """Get the latest processed frame for a camera"""
    frame_data = redis_client.get(f"crowd_frame:{cam_id}")
    if not frame_data:
        raise HTTPException(status_code=404, detail=f"No frame available for camera {cam_id}")
    return Response(content=frame_data, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)