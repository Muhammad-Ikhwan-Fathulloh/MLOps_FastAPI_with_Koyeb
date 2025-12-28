from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json
import os

app = FastAPI(title="Diabetes Prediction API", version="1.0")

# Load model
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {type(model).__name__}")
except:
    print("Model not found. Please run train_model.py first")
    model = None

# Request model
class PatientData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

# Response model
class PredictionResult(BaseModel):
    prediction: int
    probability: float
    timestamp: str

# Store predictions for monitoring
predictions_log = []

@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "retrain": "/retrain"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model else "no_model",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResult)
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert to numpy array
    features = np.array([
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]).reshape(1, -1)
    
    # Make prediction
    probability = model.predict_proba(features)[0][1]
    prediction = 1 if probability > 0.5 else 0
    
    # Log prediction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": data.dict(),
        "prediction": prediction,
        "probability": float(probability)
    }
    predictions_log.append(log_entry)
    
    # Keep only last 1000 predictions
    if len(predictions_log) > 1000:
        predictions_log.pop(0)
    
    return PredictionResult(
        prediction=prediction,
        probability=float(probability),
        timestamp=datetime.now().isoformat()
    )

@app.get("/metrics")
def get_metrics():
    """Get basic performance metrics"""
    if len(predictions_log) < 10:
        return {"message": "Not enough predictions yet", "count": len(predictions_log)}
    
    # Calculate average probability
    probabilities = [p["probability"] for p in predictions_log]
    predictions = [p["prediction"] for p in predictions_log]
    
    return {
        "total_predictions": len(predictions_log),
        "average_probability": float(np.mean(probabilities)),
        "positive_rate": float(np.mean(predictions)),
        "last_24h_count": len([p for p in predictions_log 
                              if datetime.fromisoformat(p["timestamp"]) > datetime.now().timestamp() - 86400]),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/retrain")
def retrain_model():
    """Simple retraining endpoint"""
    try:
        # In production, this would trigger a training job
        # Here we just reload the model
        global model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return {"status": "model_reloaded", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=500, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)