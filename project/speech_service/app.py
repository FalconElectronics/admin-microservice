# speech_service/app.py
from fastapi import FastAPI, HTTPException
import os
import time
import json
import asyncio
import redis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Speech and Action Analytics Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Mock data - replace with real implementation
speech_analytics_data = [
    {
        "timestamp": "23/04/18, 10:24:10",
        "Input": "Voice",
        "Output": "How are you",
        "classification": "Normal",
        "streak of Instances": "n/a",
        "confidence": "High",
        "Device Info": "Microphone 2"
    },
    {
        "timestamp": "23/04/18, 10:24:10",
        "Input": "Action",
        "Output": "Video",
        "classification": "Aggression",
        "streak of Instances": "n/a",
        "confidence": "Low",
        "Device Info": "Camera 1"
    }
]

action_analytics_data = [
    {
        "timestamp": "23/04/18, 10:24:10",
        "Input": "Voice",
        "Output": "How are you",
        "classification": "Normal",
        "streak of Instances": "n/a",
        "confidence": "High",
        "Device Info": "Microphone 2"
    },
    {
        "timestamp": "22/04/18, 12:21:09",
        "Input": "Voice",
        "Output": "Put your hands up",
        "classification": "Aggression",
        "streak of Instances": "3",
        "confidence": "High",
        "Device Info": "Microphone 1"
    }
]

@app.on_event("startup")
async def startup_event():
    # Store initial data in Redis
    redis_client.set("speech_analytics", json.dumps(speech_analytics_data))
    redis_client.set("action_analytics", json.dumps(action_analytics_data))

@app.get("/speech")
async def get_speech_analytics():
    """Get speech analytics data"""
    data = redis_client.get("speech_analytics")
    if not data:
        return []
    return json.loads(data)

@app.get("/action")
async def get_action_analytics():
    """Get action analytics data"""
    data = redis_client.get("action_analytics")
    if not data:
        return []
    return json.loads(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)