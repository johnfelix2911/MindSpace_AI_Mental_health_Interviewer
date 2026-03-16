# Deployment Guide

## Prerequisites

- Python 3.10+
- FFmpeg installed and in PATH
- ~3 GB free disk space for ML model cache
- Gemini API key

## Server Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd Mental-Health-ai-interviewer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
nano .env  # Set GEMINI_API_KEY

# 5. Download and cache all ML models (one-time)
python scripts/prepare_offline_assets.py

# 6. Verify setup
python scripts/verify_setup.py

# 7. Run
python main.py
```

## Model Downloads

The following models must be pre-cached before runtime:

| Model | Script | Size | Purpose |
|---|---|---|---|
| wav2vec2-large-xlsr-53-english | `download_ser_model.py` | ~1.2 GB | Depression SER features |
| Wav2Vec2-base (torchaudio) | `download_stress_models.py` | ~360 MB | Stress embeddings |
| StudentNet | `download_stress_models.py` | ~5 MB | Stress classifier |

Local model files (already in `models/`):
- `models/depression/phq_xgb.pkl` — XGBoost depression predictor
- `models/anxiety/gbr_pipeline.joblib` — GBR anxiety pipeline

## Production Deployment

### With Gunicorn (Linux)

```bash
pip install gunicorn
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Use `-w 1` (single worker) because the ML models are loaded in-process memory.

### With systemd

Create `/etc/systemd/system/mindspace.service`:

```ini
[Unit]
Description=MindSpace Mental Health AI
After=network.target

[Service]
User=deploy
WorkingDirectory=/opt/mindspace
ExecStart=/opt/mindspace/venv/bin/python main.py
Restart=always
Environment=GEMINI_API_KEY=your-key

[Install]
WantedBy=multi-user.target
```

### Behind Nginx

```nginx
server {
    listen 80;
    server_name mindspace.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M;
    }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8000` | Server port |
| `CORS_ORIGINS` | No | `*` | Comma-separated origins |
| `TOTAL_QUESTIONS` | No | `5` | Interview question count |
| `SER_LOAD_TIMEOUT` | No | `120` | SER model load timeout (seconds) |

## Troubleshooting

- **FFmpeg not found**: Install via `apt install ffmpeg` (Linux), `brew install ffmpeg` (Mac), or download from ffmpeg.org (Windows)
- **SER model not loading**: Run `python scripts/download_ser_model.py` and ensure you have internet access
- **Slow first inference**: First call to each pipeline loads models into memory (~10-30s). Subsequent calls are fast
- **Out of memory**: Stress + depression pipelines together need ~2-3 GB RAM. Use `CUDA_VISIBLE_DEVICES=` to force CPU if GPU memory is limited
