# 🌾 Crop Recommendation System

> AI-powered multi-model ensemble that recommends the best crop based on
> soil nutrients, climate conditions, and engineered agronomic features.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Weighted Ensemble Accuracy | **99.77 %** |
| Cohen's Kappa | **~0.998** |
| Macro F1-Score | **~0.998** |
| Classes | 22 crops |
| Features | 22 engineered |

---

## 🗂 Project Structure

```
crop-recommendation/
├── Crop_recommendation.csv          # Dataset
├── crop_recommendation_sklearn.py   # Training pipeline
├── predict_with_shap.py             # CLI prediction + SHAP
├── app.py                           # Flask web app
├── requirements.txt
├── saved_models/                    # Auto-created after training
│   ├── rf_all_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── model_weights.pkl
├── templates/
│   ├── index.html                   # Main UI
│   └── history.html                 # Prediction history
└── plots/                           # Auto-created after training
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

```bash
python crop_recommendation_sklearn.py
```

This creates `saved_models/`, `plots/`, and `performance_report.txt`.

### 3. Run the web app

```bash
python app.py
```

Visit **http://localhost:5000**

### 3.1 Enable booking confirmation emails

1. Copy `.env.example` to `.env` in the project root.
2. Fill these values in `.env`:
  - `SMTP_HOST`
  - `SMTP_PORT`
  - `SMTP_USE_TLS`
  - `SMTP_USERNAME`
  - `SMTP_PASSWORD`
  - `SMTP_FROM_EMAIL`
3. Restart the app.

Notes:
- For Gmail, use an **App Password** (not your normal account password).
- If SMTP is missing or invalid, booking is still saved but email cannot be delivered.
- Admin can verify delivery from **Admin Dashboard → Email Delivery → Send Test Email**.

### 4. Use the REST API

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"N":90,"P":42,"K":43,"temperature":21,"humidity":82,"ph":6.5,"rainfall":203}'
```

Response:
```json
{
  "success": true,
  "result": "rice",
  "confidence": 99.8,
  "alternatives": [...],
  "top_factors": [...]
}
```

### 5. CLI prediction with SHAP

```bash
python predict_with_shap.py
```

---

## 🤖 Models Used

| Model | Feature Set | Test Accuracy |
|---|---|---|
| Random Forest (tuned) | All 22 features | 99.32 % |
| Extra Trees | All 22 features | 99.09 % |
| XGBoost | All 22 features | ~99.5 % |
| LightGBM | All 22 features | ~99.4 % |
| Gradient Boosting (×2) | All / Soil+Climate | 98.64 % |
| KNN | All 22 features | 97.50 % |
| Decision Tree | Interaction features | 93.86 % |
| **Weighted Ensemble** | — | **99.77 %** |
| **Stacking Ensemble** | — | **~99.8 %** |

---

## 🔬 Feature Engineering

Beyond the raw 7 inputs (N, P, K, temperature, humidity, pH, rainfall):

- NPK ratios and sums
- Pairwise nutrient interactions
- Soil health score (weighted formula)
- Climate index
- Nutrient balance score
- pH deviation from ideal (6.5)
- Aridity index

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET/POST | `/` | Main web UI |
| POST | `/api/predict` | JSON prediction API |
| GET | `/history` | Last 50 predictions |
| GET | `/health` | Model status check |

---

## 📦 Input Ranges

| Parameter | Min | Max | Unit |
|---|---|---|---|
| Nitrogen (N) | 0 | 140 | kg/ha |
| Phosphorus (P) | 5 | 145 | kg/ha |
| Potassium (K) | 5 | 205 | kg/ha |
| Temperature | 8 | 44 | °C |
| Humidity | 14 | 100 | % |
| pH | 3.5 | 10.0 | — |
| Rainfall | 20 | 300 | mm |

---

## 🛠 Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP, Optuna
- **Web**: Flask, Flask-SQLAlchemy, SQLite
- **Viz**: Matplotlib, Seaborn
- **Explainability**: SHAP TreeExplainer

---

## 📄 Dataset

[Crop Recommendation Dataset – Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

2200 samples · 22 crop classes · 7 raw features
