# PlantScan AI — Production Deployment & Operations Guide

**Version:** 1.0.0  
**Stack:** FastAPI (Python 3.10+), TensorFlow Lite (CPU Inference), Docker, NGINX  
**Target Environments:** Cloud VM / Container Engine, Docker Compose, Local Dev, Edge Mobile  

---

## 1. Architectural Topologies

PlantScan AI supports two primary deployment topologies designed to ensure diagnostics remain available both in high-bandwidth cloud environments and low-connectivity rural farm locations:

```
                    ┌─────────────────────────┐
                    │      Client Apps        │
                    │ (Mobile / Web Frontend) │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│   Offline Edge Mode     │                 │   Cloud Inference API   │
│  - Flutter / React Nat. │                 │  - FastAPI ASGI Service │
│  - Bundled .tflite      │                 │  - Multi-crop routing   │
│  - Local classes.json   │                 │  - Thread-safe TFLite   │
│  - Embedded IPM advice  │                 │  - Full IPM DB + Trans. │
└─────────────────────────┘                 └────────────┬────────────┘
                                                         │
                                            ┌────────────┴────────────┐
                                            ▼                         ▼
                                   ┌─────────────────┐       ┌─────────────────┐
                                   │ Docker Container│       │ NGINX / Envoy   │
                                   │ (Uvicorn Worker)│       │ (Reverse Proxy) │
                                   └─────────────────┘       └─────────────────┘
```

---

## 2. Environment Configuration

All operational configuration is managed via environment variables (or `.env` file) parsed through `backend/config.py`:

| Variable | Default Value | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Deployment mode (`development`, `staging`, `production`) |
| `API_V1_PREFIX` | `/api/v1` | Versioned route mount prefix |
| `HOST` | `0.0.0.0` | Network binding interface |
| `PORT` | `8000` | Port for the ASGI server |
| `MODEL_DIR` | `ml/weights` | Absolute or relative path to TFLite model weights |
| `TREATMENT_DB_DIR` | `treatment_database` | Absolute or relative path to treatment JSON database |
| `CONFIDENCE_THRESHOLD` | `0.70` | Confidence cutoff below which predictions return `uncertain` |
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum allowable upload file size |
| `ENABLE_SIMULATION_HEADERS`| `false` | Security flag: gate `X-Simulate-*` test headers |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins (set to specific domains in prod) |

---

## 3. High-Performance Concurrency & Thread Safety

### 3.1 Non-Blocking Threadpool Execution
TFLite model inference (`interpreter.invoke()`) and PIL image decoding are CPU-bound operations. To prevent blocking the FastAPI async event loop, all inference and database parsing calls are dispatched to Starlette's threadpool:

```python
from starlette.concurrency import run_in_threadpool

# Non-blocking CPU execution:
prediction = await run_in_threadpool(
    engine.predict,
    image_array=input_array,
    crop_hint=crop_hint,
)
```

### 3.2 Thread-Safe TFLite Interpreter Invocation
`tf.lite.Interpreter` allocations and scratch buffers are **not thread-safe** across concurrent threads. PlantScan AI encloses all `interpreter.invoke()` calls inside dedicated `threading.Lock()` instances:

```python
import threading

class TFLiteModelEngine(BaseModelEngine):
    def __init__(self, ...):
        self._lock = threading.Lock()

    def predict(self, image_array, ...):
        with self._lock:
            self.interpreter.set_tensor(self.input_details[0]["index"], image_array)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output
```

---

## 4. Docker Deployment

### 4.1 Production Dockerfile
The repository includes a lean multi-stage Docker build producing a secure, unprivileged container:

```dockerfile
# Build Stage: Python 3.11-slim
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies for Pillow and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY backend/ /app/backend/
COPY ml/weights/ /app/ml/weights/
COPY treatment_database/ /app/treatment_database/

# Run as non-root user for container security
RUN useradd -m -u 1000 plantscan && \
    chown -R plantscan:plantscan /app
USER plantscan

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 4.2 Docker Compose Configuration
```yaml
version: '3.8'

services:
  plantscan-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: plantscan-backend
    restart: always
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - ENABLE_SIMULATION_HEADERS=false
      - CONFIDENCE_THRESHOLD=0.70
      - CORS_ORIGINS=["https://plantscan.ai","https://app.plantscan.ai"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2048M
```

---

## 5. Reverse Proxy Configuration (NGINX)

When terminating TLS and serving production traffic, place NGINX in front of Uvicorn:

```nginx
server {
    listen 80;
    server_name api.plantscan.ai;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.plantscan.ai;

    ssl_certificate /etc/letsencrypt/live/api.plantscan.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.plantscan.ai/privkey.pem;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Micro-caching for static treatment queries
        proxy_read_timeout 60s;
        proxy_connect_timeout 10s;
    }
}
```

---

## 6. Mobile Edge Deployment (Offline Mode)

For Flutter / React Native offline deployment:

1. **Artifact Bundling**: Copy `ml/weights/<crop>/model.tflite` and `ml/weights/<crop>/classes.json` into mobile asset folders:
   - Flutter: `assets/models/<crop>/` (registered in `pubspec.yaml`).
   - React Native: `android/app/src/main/assets/models/<crop>/`.
2. **Offline IPM Advisories**: Include `treatment_database/translations/*.json` and crop JSONs in local SQLite or bundled JSON assets.
3. **Interpreter Management**: Instantiate a singleton interpreter per crop; re-use across camera frames to avoid memory reallocations.

---

## 7. Health Checks & Observability

- **Liveness & Readiness**: Query `GET /api/v1/health`.
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "models_loaded": {
      "tomato": true,
      "grape": true,
      "apple": true
    },
    "treatment_database_records": 46,
    "supported_languages": ["en", "hi", "kn"]
  }
  ```
- **Latency Monitoring**: Each diagnostic response returns `inference_time_ms` capturing raw neural network inference and processing latency.
