from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import numpy as np
from utils.donwload_model import download_model

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok 🚀"}

# jalankan sebelum load model
download_model()

# ===============================
# IMPORT PIPELINE ML
# ===============================
from services.inference import run_inference

app = FastAPI(title="Machine Diagnostics API")

# ===============================
# CORS (WAJIB untuk React)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: ganti domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

# ===============================
# ANALYZE AUDIO (ML AKTIF)
# ===============================
@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    mode: str = Form(...)
):
    """
    Terima audio dari frontend (audio/webm)
    Mode: quick / deep
    """

    # ===============================
    # READ AUDIO
    # ===============================
    audio_bytes = await file.read()

    # ===============================
    # ML INFERENCE
    # ===============================
    label, confidence = run_inference(audio_bytes)
    confidence = float(confidence)

    # ===============================
    # MAP MODEL OUTPUT → ISSUE
    # (CONTOH: binary classification)
    # ===============================
    issues = []

    if label == 1:
        issues.append({
            "severity": "high",
            "component": "Bearing Motor",
            "description": "Anomali suara mesin terdeteksi oleh model AI",
            "recommendation": "Segera lakukan inspeksi dan penggantian bearing"
        })

    # ===============================
    # HEALTH SCORE
    # ===============================
    overall_health = max(0, min(100, int(100 - confidence * 100)))

    # ===============================
    # VIBRATION DATA (UNTUK VISUAL)
    # ===============================
    points = 100 if mode == "quick" else 300
    vibration_data = [
        {
            "time": round(i * 0.01, 4),
            "amplitude": float(np.sin(i * 0.1)),
            "frequency": float(60 + np.sin(i * 0.05) * 60)
        }
        for i in range(points)
    ]

    return {
        "overall_health": overall_health,
        "issues": issues,
        "vibration_data": vibration_data
    }

# ===============================
# HISTORY (DUMMY / READY FOR DB)
# ===============================
@app.get("/api/history")
async def get_history(limit: int = 10):
    return {
        "records": []
    }