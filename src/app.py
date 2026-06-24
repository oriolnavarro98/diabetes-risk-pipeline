"""
app.py - route, validate, delegate, respond based on the API calls
Run from project root: uvicorn src.app:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse
from .predict import run_predict, load_artifact

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifact()
    yield

app = FastAPI(title="ML Diabetes risk predictor", lifespan=lifespan)

@app.get("/health")
def health():
    _, metadata = load_artifact()
    return {"status": "ok", "model_version": metadata["version"]}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        return run_predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))