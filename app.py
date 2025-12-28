from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List

app = FastAPI(title="Diabetes API", version="1.0")

# Load model
try:
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    print(f"Model loaded with {len(features)} features")
except:
    print("Warning: Using fallback model")
    model = None
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

class Patient(BaseModel):
    pregnancies: int = 1
    glucose: float = 100.0
    blood_pressure: float = 70.0
    skin_thickness: float = 20.0
    insulin: float = 80.0
    bmi: float = 25.0
    diabetes_pedigree: float = 0.5
    age: int = 30

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: Patient):
    # Prepare features
    X = np.array([
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.diabetes_pedigree,
        patient.age
    ]).reshape(1, -1)
    
    if model:
        proba = model.predict_proba(X)[0]
        prediction = int(proba[1] > 0.5)
        probability = float(proba[1])
    else:
        # Fallback calculation
        risk_score = (
            patient.glucose / 200 * 0.4 +
            patient.bmi / 50 * 0.3 +
            patient.age / 100 * 0.2 +
            patient.pregnancies / 10 * 0.1
        )
        probability = min(max(risk_score, 0), 1)
        prediction = 1 if probability > 0.5 else 0
    
    return {
        "prediction": prediction,
        "probability": probability,
        "risk": "high" if probability > 0.7 else "medium" if probability > 0.3 else "low"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)