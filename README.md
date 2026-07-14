#  Plant Disease Detection — Deep Learning Model

A production-ready deep learning pipeline for detecting plant diseases from leaf images, using **EfficientNetB0** transfer learning. Supports 38 disease classes across 14 crop species.

---

##  Project Structure

```
plant-disease-detection/
├── plant_disease_colab.ipynb   ← Full training script (run on Google Colab)
├── app.py                   ← Flask web app for inference
├── requirements.txt         ← Python dependencies
├── Dockerfile               ← Docker container for cloud deployment
└── README.md
```

---

##  Quick Start

### Step 1 — Train on Google Colab (Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Upload `plant_disease_colab.py` or copy-paste cells
4. Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
5. Download dataset:
   ```python
   # Upload your kaggle.json first, then:
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d emmarex/plantdisease --unzip -p /content/data
   ```
6. Run all cells — training takes ~30–60 min on T4

### Step 2 — Run the Web App Locally

```bash
pip install -r requirements.txt
python app.py
# → Open http://localhost:8080
```

---

##  Cloud Deployment Options

### Option A — Render (Free Tier, Recommended)
1. Push this project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT app:app`
   - **Environment Variables:** `MODEL_PATH`, `CLASS_NAMES_PATH`
5. Upload model files via Render Disk or store in Google Cloud Storage

### Option B — Railway
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Option C — Hugging Face Spaces (Best for ML Models)
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Gradio** SDK
3. Upload model + `app.py` adapted for Gradio:
   ```python
   import gradio as gr
   # wrap predict() in a gr.Interface
   ```

### Option D — Docker (Any Cloud)
```bash
# Build
docker build -t plant-disease-app .

# Run locally
docker run -p 8080:8080 \
  -v $(pwd)/model:/app/model \
  -e MODEL_PATH=/app/model/plant_disease_model.keras \
  -e CLASS_NAMES_PATH=/app/model/class_names.json \
  plant-disease-app

# Push to Google Cloud Run
docker tag plant-disease-app gcr.io/YOUR_PROJECT/plant-disease-app
docker push gcr.io/YOUR_PROJECT/plant-disease-app
gcloud run deploy --image gcr.io/YOUR_PROJECT/plant-disease-app --platform managed
```

### Option E — AWS Lambda (Serverless)
Use `serverless-python-requirements` + API Gateway for low-cost serverless inference.

---

##  Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 Backbone (pretrained on ImageNet, 237 layers)
    ↓  [Phase 1: frozen | Phase 2: top layers unfrozen]
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, ReLU) → Dropout(0.4)
    ↓
Dense(256, ReLU) → Dropout(0.2)
    ↓
Dense(38, Softmax)
    ↓
Output: Disease class probabilities
```

### Training Strategy
| Phase | Layers trained | LR       | Epochs |
|-------|---------------|----------|--------|
| 1     | Head only     | 1e-3     | 20     |
| 2     | Top backbone  | 1e-5     | 30-50  |

---

##  Supported Diseases (38 Classes)

| Crop       | Diseases Detected |
|------------|-------------------|
| Tomato     | Late Blight, Early Blight, Leaf Mold, Mosaic Virus, Yellow Leaf Curl, Septoria Leaf Spot, Spider Mites, Bacterial Spot, Target Spot, Healthy |
| Apple      | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Potato     | Early Blight, Late Blight, Healthy |
| Corn       | Common Rust, Northern Leaf Blight, Cercospora Leaf Spot, Healthy |
| Grape      | Black Rot, Esca, Leaf Blight, Healthy |
| Pepper     | Bacterial Spot, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| Peach      | Bacterial Spot, Healthy |
| Cherry     | Powdery Mildew, Healthy |
| Soybean    | Healthy |
| Squash     | Powdery Mildew |
| Raspberry  | Healthy |
| Orange     | Citrus Greening |

---

##  Expected Performance

| Metric        | Expected Score |
|---------------|---------------|
| Val Accuracy  | 95–98%        |
| Top-3 Acc     | 99%+          |
| F1-Score (avg)| 0.95–0.97     |

*(Results on PlantVillage dataset with EfficientNetB0)*

---

##  API Endpoints

| Endpoint  | Method | Description               |
|-----------|--------|---------------------------|
| `/`       | GET    | Web UI                    |
| `/predict`| POST   | Upload image, get results |
| `/health` | GET    | Model health check        |

### `/predict` – Example Response
```json
{
  "status": "success",
  "top_disease": "Tomato___Late_blight",
  "confidence": "97.34%",
  "predictions": [
    {
      "rank": 1,
      "disease": "Tomato___Late_blight",
      "plant": "Tomato",
      "condition": "Late blight",
      "confidence": 0.9734,
      "confidence_pct": "97.34%"
    }
  ]
}
```

---

##  Dataset

- **Primary:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Alternative:** [TF Datasets plant_village](https://www.tensorflow.org/datasets/catalog/plant_village)
- **Size:** ~54,000 images, 38 classes

---

##  Requirements

- Python 3.9+
- TensorFlow 2.13+
- 8 GB+ RAM (16 GB recommended)
- GPU recommended (Google Colab T4 works perfectly)
