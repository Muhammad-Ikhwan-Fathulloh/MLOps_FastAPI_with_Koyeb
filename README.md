# Simple Diabetes Prediction API

Production-ready ML API for diabetes prediction using Pima Indians dataset.

## Features
- Predict diabetes probability
- Basic metrics monitoring
- Model retraining endpoint
- Docker & Koyeb ready

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run API
python app.py

# Or with uvicorn
uvicorn app:app --reload
```