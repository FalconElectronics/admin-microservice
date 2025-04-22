# api_gateway/app.py
from fastapi import FastAPI, Depends, HTTPException
import httpx
import os

app = FastAPI(title="Inference API Gateway")

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth:5000")
WEAPON_SERVICE_URL = os.getenv("WEAPON_SERVICE_URL", "http://weapon:5001")
CROWD_SERVICE_URL = os.getenv("CROWD_SERVICE_URL", "http://crowd:5002")
SPEECH_SERVICE_URL = os.getenv("SPEECH_SERVICE_URL", "http://speech:5003")

async def verify_token(token: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{AUTH_SERVICE_URL}/verify", json={"token": token})
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        return response.json()

@app.get("/")
async def root():
    return {"message": "Inference API Gateway"}

@app.get("/weapon-detection")
async def weapon_detection(token: str):
    user = await verify_token(token)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{WEAPON_SERVICE_URL}/detection", headers={"Authorization": f"Bearer {token}"})
        return response.json()

@app.get("/crowd-analytics")
async def crowd_analytics(token: str):
    user = await verify_token(token)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CROWD_SERVICE_URL}/analytics", headers={"Authorization": f"Bearer {token}"})
        return response.json()

@app.get("/speech-analytics")
async def speech_analytics(token: str):
    user = await verify_token(token)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SPEECH_SERVICE_URL}/analytics", headers={"Authorization": f"Bearer {token}"})
        return response.json()