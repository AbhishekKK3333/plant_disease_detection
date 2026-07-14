# ══════════════════════════════════════════════════════════════
#  Plant Disease Detection – Production Docker Image
#  Build : docker build -t plant-disease-app .
#  Run   : docker run -p 8080:8080 \
#            -v /path/to/model:/app/model \
#            -e MODEL_PATH=/app/model/plant_disease_model.keras \
#            -e CLASS_NAMES_PATH=/app/model/class_names.json \
#            plant-disease-app
# ══════════════════════════════════════════════════════════════

FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py .

# Environment defaults
ENV MODEL_PATH=plant_disease_model.keras
ENV CLASS_NAMES_PATH=class_names.json
ENV IMG_SIZE=224
ENV PORT=8080

EXPOSE 8080

# Production server via gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app:app"]
