# weapon_service/app.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
import os
import cv2
import time
import json
from datetime import datetime
import torch
from ultralytics import YOLO
import asyncio
import redis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Weapon Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for storing detection results
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Load the weapon detection model
model = YOLO(os.getenv("WEAPON_MODEL_PATH", "best_yolo11x_gun.engine"))

# Output directory for detections
output_dir = os.getenv("OUTPUT_DIR", "detections")
os.makedirs(output_dir, exist_ok=True)

# Store detection data
detection_data = {}

async def process_frame(frame, cam_id):
    """Process a single frame for weapon detection"""
    results = model(frame)
    detected = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            if class_id == 0 and conf > 0.60:  # Handgun detection with confidence > 60%
                detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Handgun {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if detected:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = f"{output_dir}/{cam_id}_detection_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Save detection data to redis
        detection_event = {
            "timestamp": timestamp,
            "camera": cam_id,
            "image_path": image_path
        }
        
        cam_detections = detection_data.get(cam_id, [])
        cam_detections.append(detection_event)
        detection_data[cam_id] = cam_detections
        
        # Store in Redis for other services to access
        redis_client.hset("weapon_detections", cam_id, json.dumps(detection_event))
        
    return frame, detected

async def continuous_detection(camera_sources):
    """Continuously process frames from multiple camera sources"""
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
            
            processed_frame, detected = await process_frame(frame, cam_id)
            
            # Store the processed frame for streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            redis_client.set(f"frame:{cam_id}", buffer.tobytes())
            
            if detected:
                print(f"[ALERT] Weapon detected on camera {cam_id}")
        
        await asyncio.sleep(0.03)  # Throttle processing

@app.on_event("startup")
async def startup_event():
    # Define camera sources - in production, load from config
    camera_sources = {
        "cam1": os.getenv("CAMERA1_URL", "rtsp://192.168.1.121:554/stream0"),
        "cam2": os.getenv("CAMERA2_URL", "rtsp://192.168.1.30:554/stream0"),
        "cam3": os.getenv("CAMERA3_URL", "rtsp://192.168.1.27:554/stream0")
    }
    
    # Start continuous detection in background
    asyncio.create_task(continuous_detection(camera_sources))

@app.get("/detection")
async def get_detections():
    """Get all weapon detection events"""
    return detection_data

@app.get("/detection/{cam_id}")
async def get_camera_detections(cam_id: str):
    """Get weapon detection events for a specific camera"""
    return detection_data.get(cam_id, [])

@app.get("/frame/{cam_id}")
async def get_latest_frame(cam_id: str):
    """Get the latest processed frame for a camera"""
    frame_data = redis_client.get(f"frame:{cam_id}")
    if not frame_data:
        raise HTTPException(status_code=404, detail=f"No frame available for camera {cam_id}")
    return Response(content=frame_data, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)