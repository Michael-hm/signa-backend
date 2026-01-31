from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import json
import os
from pathlib import Path
from keras.layers import TFSMLayer
from keras import Model , Input
from typing import List


# -------------------------
# Cargar artifacts
# -------------------------

# Ruta absoluta al directorio actual 
BASE_DIR = Path(__file__).resolve().parent

# Ruta al directorio artifacts
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "signa_model"
LABELS_PATH = ARTIFACTS_DIR / "idx_to_label.json"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

WINDOW_SIZE = int(cfg["window_size"])
FEATURES = int(cfg["features"])

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    idx_to_label = {int(k): v for k, v in json.load(f).items()}
    

layer = TFSMLayer(
    str(MODEL_PATH),
    call_endpoint="serving_default"
)

inputs = Input(shape=(WINDOW_SIZE, FEATURES))
outputs = layer(inputs)

model = Model(inputs=inputs, outputs=outputs)

# -------------------------
# API
# -------------------------
app = FastAPI(title="SIGNA Backend", version="1.0")

# CORS: para que luego Vercel (otro dominio) pueda llamar a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en demo OK; en producción pondremos tu dominio de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Lista de 40 frames, cada frame con 63 valores float"
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

@app.post("/predict_sequence")
def predict_sequence(payload: PredictRequest):
    seq = np.array(payload.sequence, dtype=np.float32)

    if seq.shape != (WINDOW_SIZE, FEATURES):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input shape",
                "expected": [WINDOW_SIZE, FEATURES],
                "received": list(seq.shape),
            },
        )

    x = np.expand_dims(seq, axis=0)  # (1, 40, 63)

    raw_preds = model.predict(x, verbose=0)

    if isinstance(raw_preds, dict):
        preds = list(raw_preds.values())[0][0]
    else:
        preds = raw_preds[0]

    idx = int(np.argmax(preds))
    label = idx_to_label.get(idx, "unknown")
    confidence = float(preds[idx])

    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [
        {
            "label": idx_to_label.get(int(i), "unknown"),
            "confidence": float(preds[int(i)]),
        }
        for i in top3_idx
    ]

    return {
        "label": label,
        "confidence": confidence,
        "top3": top3,
    }