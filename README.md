# MLOps Diabetes Prediction API

Sebuah API sederhana untuk prediksi diabetes menggunakan dataset Pima Indians yang siap di-deploy ke production dengan Koyeb.

## 📋 Fitur Utama

- ✅ **Prediksi Real-time** - Endpoint POST untuk prediksi diabetes
- ✅ **Health Monitoring** - Endpoint health check
- ✅ **Performance Metrics** - Monitoring performa model
- ✅ **Model Retraining** - Endpoint untuk retrain model
- ✅ **Docker Support** - Containerized application
- ✅ **Koyeb Ready** - Siap deploy ke Koyeb dengan satu klik
- ✅ **Dataset Real** - Menggunakan Pima Indians Diabetes Database dari Kaggle

## 🏗️ Arsitektur

```
┌─────────────────┐
│   Client App    │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (Python 3.9)   │
├─────────────────┤
│  /predict       │
│  /health        │
│  /metrics       │
│  /retrain       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│  Random Forest  │
└─────────────────┘
```

## 🚀 Deployment Quick Start

### Metode 1: Deploy ke Koyeb (Rekomendasi)

1. **Fork/Push ke GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Diabetes Prediction API"
   git branch -M main
   git remote add origin https://github.com/username/mlops-diabetes.git
   git push -u origin main
   ```

2. **Deploy di Koyeb**
   - Buka [Koyeb Console](https://app.koyeb.com)
   - Klik "Create App"
   - Pilih "GitHub" sebagai sumber
   - Pilih repository Anda
   - Koyeb akan auto-detect Dockerfile
   - Klik "Deploy"

3. **Akses API**
   ```
   https://your-app-name.koyeb.app
   ```

### Metode 2: Deploy dengan Docker

```bash
# Build image
docker build -t diabetes-api .

# Run container
docker run -p 8000:8000 diabetes-api

# Or with docker-compose
docker-compose up
```

### Metode 3: Run Lokal

```bash
# 1. Clone repository
git clone https://github.com/username/mlops-diabetes.git
cd mlops-diabetes

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model awal
python train_model.py

# 4. Jalankan server
python app.py
# atau
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Dataset

API ini menggunakan **Pima Indians Diabetes Database** dari Kaggle:

- **Sumber**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Samples**: 768 records
- **Features**: 8 medical attributes
- **Target**: Binary (1=Diabetes, 0=Non-Diabetes)

**Features:**
1. Pregnancies - Jumlah kehamilan
2. Glucose - Konsentrasi glukosa plasma
3. BloodPressure - Tekanan darah diastolik (mm Hg)
4. SkinThickness - Ketebalan lipatan kulit trisep (mm)
5. Insulin - Insulin serum 2-hour (mu U/ml)
6. BMI - Body mass index (kg/m²)
7. DiabetesPedigreeFunction - Fungsi pedigree diabetes
8. Age - Umur (tahun)

## 🔧 API Endpoints

### 1. `GET /` - Root Endpoint
```bash
curl https://your-app.koyeb.app/
```
**Response:**
```json
{
  "message": "Diabetes Prediction API",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "metrics": "/metrics",
    "retrain": "/retrain"
  }
}
```

### 2. `GET /health` - Health Check
```bash
curl https://your-app.koyeb.app/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3. `POST /predict` - Prediksi Diabetes
```bash
curl -X POST "https://your-app.koyeb.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
  }'
```
**Response:**
```json
{
  "prediction": 1,
  "probability": 0.85,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 4. `GET /metrics` - Performance Metrics
```bash
curl https://your-app.koyeb.app/metrics
```
**Response:**
```json
{
  "total_predictions": 150,
  "average_probability": 0.42,
  "positive_rate": 0.35,
  "last_24h_count": 25,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 5. `POST /retrain` - Retrain Model
```bash
curl -X POST "https://your-app.koyeb.app/retrain"
```
**Response:**
```json
{
  "status": "model_reloaded",
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python train_model.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
# Build image
docker build -t diabetes-api .

# Run container
docker run -d \
  -p 8000:8000 \
  --name diabetes-api \
  diabetes-api

# Check logs
docker logs diabetes-api

# Stop container
docker stop diabetes-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🌐 Koyeb Configuration

### Environment Variables (Opsional)
```bash
# Di Koyeb dashboard -> Settings -> Environment Variables
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

### Scaling (Opsional)
- **Instances**: 1-3 (sesuai kebutuhan)
- **Memory**: 512MB - 1GB
- **CPU**: 0.25 - 0.5 vCPU

### Custom Domain (Opsional)
1. Di Koyeb dashboard, pilih app Anda
2. Klik "Settings" → "Domains"
3. Tambahkan custom domain
4. Setup DNS records sesuai petunjuk

## 🧪 Testing

### Test dengan Python
```python
import requests
import json

url = "https://your-app.koyeb.app/predict"

data = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 80,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 26.5,
    "diabetes_pedigree": 0.4,
    "age": 30
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

### Test dengan cURL
```bash
# Test semua endpoints
ENDPOINT="https://your-app.koyeb.app"

echo "1. Health check:"
curl "$ENDPOINT/health"

echo "\n2. Make prediction:"
curl -X POST "$ENDPOINT/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 1,
    "glucose": 85,
    "blood_pressure": 66,
    "skin_thickness": 29,
    "insulin": 0,
    "bmi": 26.6,
    "diabetes_pedigree": 0.351,
    "age": 31
  }'

echo "\n3. Get metrics:"
curl "$ENDPOINT/metrics"
```

## 📈 Monitoring & Maintenance

### 1. Health Monitoring
- Endpoint `/health` untuk monitoring uptime
- Bisa diintegrasikan dengan Koyeb health checks
- Auto-restart jika unhealthy

### 2. Performance Tracking
- Endpoint `/metrics` memberikan stats performa
- Track positive rate untuk drift detection
- Monitor prediction volume

### 3. Model Maintenance
- Retrain model secara berkala dengan `/retrain`
- Monitor accuracy degradation
- Version control untuk model

### 4. Logs Monitoring
```bash
# Koyeb logs
koyeb app logs your-app-name

# Docker logs
docker logs diabetes-api
```

## 🔐 Security Considerations

### 1. Rate Limiting (Rekomendasi)
Tambahkan di Koyeb:
```yaml
# Di Koyeb Advanced Settings
rate_limit:
  requests: 100
  window: 60s
```

### 2. API Key Authentication (Opsional)
```python
# Tambahkan di app.py
API_KEYS = {"client1": "secret-key-123"}

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path not in ["/", "/health"]:
        api_key = request.headers.get("X-API-Key")
        if api_key not in API_KEYS.values():
            return JSONResponse({"error": "Invalid API key"}, status_code=401)
    return await call_next(request)
```

### 3. CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain spesifik di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚨 Troubleshooting

### Issue: Model tidak load
```bash
# Solution: Train model dulu
python train_model.py

# Pastikan model.pkl ada
ls -la model.pkl
```

### Issue: Port sudah digunakan
```bash
# Cari process yang menggunakan port 8000
lsof -i :8000

# Kill process atau ganti port
uvicorn app:app --port 8001
```

### Issue: Kaggle dataset gagal download
```bash
# Manual download:
# 1. Download dari https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# 2. Simpan sebagai diabetes.csv di folder yang sama
# 3. Modifikasi train_model.py untuk load file lokal
```

### Issue: Docker build error
```bash
# Clear Docker cache
docker system prune -a

# Build tanpa cache
docker build --no-cache -t diabetes-api .
```

## 📚 Code Structure

```
mlops-diabetes/
├── app.py                    # FastAPI application
├── train_model.py           # Model training script
├── model.pkl               # Trained model (generated)
├── features.pkl            # Feature names (generated)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── .dockerignore          # Docker ignore file
├── docker-compose.yml     # Docker compose (opsional)
└── README.md             # This file
```

### File Descriptions:
- **app.py**: FastAPI server dengan semua endpoints
- **train_model.py**: Download dataset dan train model
- **requirements.txt**: Python package dependencies
- **Dockerfile**: Konfigurasi Docker container
- **.dockerignore**: File yang diignore oleh Docker

## 🔄 CI/CD Pipeline (Opsional)

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Koyeb

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Koyeb
        uses: koyeb/action-deploy@v1
        with:
          koyeb_token: ${{ secrets.KOYEB_TOKEN }}
          app_name: diabetes-api
          service_type: web
```

## 🎯 Use Cases

### 1. Aplikasi Kesehatan
```javascript
// Frontend integration example
async function checkDiabetes(patientData) {
  const response = await fetch('https://your-app.koyeb.app/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patientData)
  });
  
  const result = await response.json();
  return result.prediction === 1 ? 'High Risk' : 'Low Risk';
}
```

### 2. Batch Processing
```python
# Process multiple patients
import pandas as pd
import requests

df = pd.read_csv('patients.csv')
results = []

for _, row in df.iterrows():
    response = requests.post('https://your-app.koyeb.app/predict', json=row.to_dict())
    results.append(response.json())

pd.DataFrame(results).to_csv('predictions.csv')
```

### 3. Monitoring Dashboard
```python
# Simple monitoring script
import requests
import time
from datetime import datetime

while True:
    # Check health
    health = requests.get('https://your-app.koyeb.app/health').json()
    
    # Get metrics
    metrics = requests.get('https://your-app.koyeb.app/metrics').json()
    
    print(f"[{datetime.now()}] Health: {health['status']}, Predictions: {metrics['total_predictions']}")
    
    time.sleep(300)  # Check every 5 minutes
```

## 📞 Support

### 1. API Issues
- Check health endpoint: `GET /health`
- Check server logs di Koyeb dashboard
- Verify model is loaded (`model_loaded: true`)

### 2. Deployment Issues
- Verify Dockerfile syntax
- Check Koyeb build logs
- Ensure GitHub repository is public/accessible

### 3. Model Issues
- Retrain model dengan `POST /retrain`
- Verify dataset availability
- Check feature dimensions match

### 4. Performance Issues
- Scale up di Koyeb dashboard
- Add caching layer
- Optimize model size

## 📄 License

MIT License - feel free to use for personal and commercial projects.

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Database from UCI ML Repository
- FastAPI: Modern web framework for Python
- Koyeb: Serverless platform for deployment
- Scikit-learn: Machine learning library

---

**Deploy Now:** [![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/apps/deploy?type=docker&image=yourusername/diabetes-api&name=diabetes-api)

**API Documentation:** Available at `https://your-app.koyeb.app/docs` (auto-generated by FastAPI)

**Example Application:** [Live Demo](https://diabetes-api-example.koyeb.app) (if deployed)

---

<div align="center">
  <p>Made with ❤️ for MLOps Production Systems</p>
  <p>Happy Coding! 🚀</p>
</div>