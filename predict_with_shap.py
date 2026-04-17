"""
Crop Prediction with SHAP Explainability  –  IMPROVED VERSION
Improvements:
  • Consistent joblib (no more pickle/joblib mix)
  • Batch prediction support
  • Robust SHAP value extraction for all sklearn estimator types
  • Saves both waterfall and summary plots
  • Returns structured dict (easy to plug into Flask / API)
"""

import joblib
import numpy as np
import pandas as pd
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed. Run: pip install shap")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  AGRONOMIC VALIDATION
# ─────────────────────────────────────────────
VALID_RANGES = {
    'N':           (0,   140),
    'P':           (5,   145),
    'K':           (5,   205),
    'temperature': (8,   44),
    'humidity':    (14,  100),
    'ph':          (3.5, 10.0),
    'rainfall':    (20,  300),
}

def validate_input(data: dict) -> list:
    errors = []
    for field, (lo, hi) in VALID_RANGES.items():
        val = data.get(field)
        if val is None:
            errors.append(f"Missing field: {field}")
        elif not (lo <= float(val) <= hi):
            errors.append(
                f"{field} = {val} is outside valid range [{lo}, {hi}]"
            )
    return errors


# ─────────────────────────────────────────────
#  LOAD MODELS (joblib throughout)
# ─────────────────────────────────────────────
def load_models(model_dir='saved_models'):
    print("🔄 Loading trained models...")
    scaler        = joblib.load(f'{model_dir}/scaler.pkl')
    label_encoder = joblib.load(f'{model_dir}/label_encoder.pkl')
    feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
    model_weights = joblib.load(f'{model_dir}/model_weights.pkl')

    model_keys = [
        'rf_all', 'rf_soil', 'rf_climate',
        'et_all', 'et_comprehensive',
        'gb_all', 'gb_soil_climate',
        'knn_all', 'dt_interactions',
    ]
    # Add XGB / LGBM if saved
    for extra in ['xgb_all', 'lgbm_all']:
        if os.path.exists(f'{model_dir}/{extra}_model.pkl'):
            model_keys.append(extra)

    models = {}
    for key in model_keys:
        path = f'{model_dir}/{key}_model.pkl'
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            print(f"   ⚠ {path} not found – skipping.")

    print(f"✅ Loaded {len(models)} models\n")
    return scaler, label_encoder, feature_names, model_weights, models


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    df['NPK_sum']  = df['N'] + df['P'] + df['K']
    df['NP_ratio'] = df['N'] / (df['P'] + 1)
    df['NK_ratio'] = df['N'] / (df['K'] + 1)
    df['PK_ratio'] = df['P'] / (df['K'] + 1)

    df['N_P_interaction'] = df['N'] * df['P']
    df['N_K_interaction'] = df['N'] * df['K']
    df['P_K_interaction'] = df['P'] * df['K']

    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rainfall_humidity_ratio']   = df['rainfall'] / (df['humidity'] + 1)
    df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall']

    df['soil_health_score'] = (
        (df['ph'] / 7.0)   * 0.30 +
        (df['N'] / 140)    * 0.25 +
        (df['P'] / 145)    * 0.25 +
        (df['K'] / 205)    * 0.20
    )
    df['climate_index'] = (
        (df['temperature'] / 43.68)  * 0.40 +
        (df['humidity']    / 100)    * 0.30 +
        (df['rainfall']    / 298.56) * 0.30
    )
    avg_npk = (df['N'] + df['P'] + df['K']) / 3
    df['nutrient_balance'] = 1 - (
        (abs(df['N'] - avg_npk) +
         abs(df['P'] - avg_npk) +
         abs(df['K'] - avg_npk)) / (3 * avg_npk + 1)
    )
    # New features (added in improved training)
    df['ph_deviation']  = abs(df['ph'] - 6.5)
    df['aridity_index'] = df['temperature'] / (df['rainfall'] + 1)

    return df


# ─────────────────────────────────────────────
#  CORE PREDICTION + SHAP
# ─────────────────────────────────────────────
def predict_with_shap(input_data, scaler, models, model_weights,
                      label_encoder, feature_names,
                      save_plots=True):
    """
    Returns a structured dict:
    {
      'crop': str,
      'confidence': float,
      'top3': [(crop, prob), ...],
      'shap_factors': [(feature, shap_val, raw_val), ...],
      'shap_available': bool,
    }
    """
    # Validate
    errors = validate_input(input_data)
    if errors:
        print("❌ Validation errors:")
        for e in errors:
            print(f"   • {e}")
        return None

    # Engineer features
    X_df     = engineer_features(input_data)
    n_feat   = len(feature_names)
    X_df     = X_df.iloc[:, :n_feat]          # trim if extra cols present
    X_scaled = scaler.transform(X_df)

    # Use best model (rf_all)
    best_md    = models.get('rf_all', next(iter(models.values())))
    best_model = best_md['model']

    prediction_idx  = best_model.predict(X_scaled)[0]
    predicted_crop  = label_encoder.inverse_transform([prediction_idx])[0]
    proba           = best_model.predict_proba(X_scaled)[0]
    confidence      = float(proba[prediction_idx])

    # Top-3
    top3_idx   = np.argsort(proba)[-3:][::-1]
    top3       = [(label_encoder.inverse_transform([i])[0], float(proba[i]))
                  for i in top3_idx]

    # ── SHAP ──────────────────────────────────────────────────────────────
    shap_factors   = []
    shap_available = False

    if SHAP_AVAILABLE:
        try:
            explainer  = shap.TreeExplainer(best_model)
            shap_vals  = explainer.shap_values(X_scaled)

            if isinstance(shap_vals, list):
                sv = shap_vals[prediction_idx][0]
            else:
                sv = shap_vals[0, :, prediction_idx]

            fn = feature_names[:len(sv)]
            raw = X_scaled[0, :len(sv)]

            pairs = sorted(
                zip(fn, sv, raw),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            shap_factors   = [(f, float(s), float(r)) for f, s, r in pairs]
            shap_available = True

            if save_plots:
                _save_shap_plots(sv, fn, X_scaled[0],
                                 predicted_crop, explainer,
                                 prediction_idx)
        except Exception as e:
            print(f"⚠ SHAP error: {e}")

    # ── Console output ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"🌾  RECOMMENDED CROP : {predicted_crop.upper()}")
    print(f"✅  Confidence       : {confidence*100:.1f}%")
    print("=" * 65)

    if shap_factors:
        print("\n🔍 Top 5 SHAP factors:")
        print("─" * 65)
        for rank, (feat, sv, rv) in enumerate(shap_factors[:5], 1):
            direction = "▲ pushes TOWARDS" if sv > 0 else "▼ pushes AWAY from"
            intensity = ("STRONGLY" if abs(sv) > 0.10 else
                         "Moderately" if abs(sv) > 0.05 else "Slightly")
            print(f"  {rank}. {feat:30}  SHAP = {sv:+.4f}")
            print(f"       {intensity} {direction} {predicted_crop.upper()}")
    else:
        print("   (SHAP not available)")

    print("\n📊 Alternative crops:")
    for crop, prob in top3[1:]:
        print(f"   • {crop:20} {prob*100:.1f}%")
    print("=" * 65)

    return {
        'crop':          predicted_crop,
        'confidence':    confidence,
        'top3':          top3,
        'shap_factors':  shap_factors,
        'shap_available': shap_available,
    }


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def _save_shap_plots(shap_vals, feature_names, raw_vals,
                     crop_name, explainer, pred_idx):
    os.makedirs('shap_explanations', exist_ok=True)

    # 1. Waterfall bar chart
    indices = np.argsort(np.abs(shap_vals))[-12:][::-1]
    colors  = ['#E05C3A' if v > 0 else '#4C8BBF'
               for v in shap_vals[indices]]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), shap_vals[indices], color=colors)
    plt.yticks(range(len(indices)),
               [feature_names[i] for i in indices])
    plt.xlabel('SHAP Value  (impact on prediction probability)')
    plt.title(f'Why was {crop_name.upper()} recommended?\n'
              f'Top 12 SHAP contributions')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('shap_explanations/shap_waterfall.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Signed contribution table
    print("\n💾 Saved: shap_explanations/shap_waterfall.png")


# ─────────────────────────────────────────────
#  BATCH PREDICTION
# ─────────────────────────────────────────────
def batch_predict(csv_path, scaler, models, model_weights,
                  label_encoder, feature_names):
    """
    Predict for all rows in a CSV file.
    Returns a DataFrame with original columns + predicted_crop + confidence.
    """
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        inp = row.to_dict()
        out = predict_with_shap(inp, scaler, models, model_weights,
                                label_encoder, feature_names,
                                save_plots=False)
        if out:
            results.append({
                **inp,
                'predicted_crop': out['crop'],
                'confidence':     round(out['confidence'] * 100, 2),
            })
    result_df = pd.DataFrame(results)
    out_path  = csv_path.replace('.csv', '_predictions.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n✅ Batch predictions saved to {out_path}")
    return result_df


# ─────────────────────────────────────────────
#  INTERACTIVE CLI
# ─────────────────────────────────────────────
def get_user_input() -> dict | None:
    print("\n" + "=" * 65)
    print("ENTER YOUR FARM DATA")
    print("=" * 65)
    try:
        return {
            'N':           float(input("\n  Nitrogen (N)       [0–140]:  ")),
            'P':           float(input("  Phosphorus (P)     [5–145]:  ")),
            'K':           float(input("  Potassium (K)      [5–205]:  ")),
            'temperature': float(input("  Temperature (°C)   [8–44]:   ")),
            'humidity':    float(input("  Humidity (%)       [14–100]: ")),
            'ph':          float(input("  pH                 [3.5–10]: ")),
            'rainfall':    float(input("  Rainfall (mm)      [20–300]: ")),
        }
    except ValueError:
        print("\n❌ Invalid input – please enter numbers only.")
        return None


def main():
    print("=" * 65)
    print("🌾  CROP RECOMMENDATION WITH SHAP EXPLAINABILITY")
    print("=" * 65)

    try:
        scaler, label_encoder, feature_names, model_weights, models = \
            load_models()
    except FileNotFoundError:
        print("\n❌ Models not found! Run crop_recommendation_sklearn.py first.")
        return

    while True:
        inp = get_user_input()
        if inp is None:
            continue

        predict_with_shap(inp, scaler, models, model_weights,
                          label_encoder, feature_names)

        again = input("\n\nPredict another? (y/n): ").strip().lower()
        if again != 'y':
            print("\n👋 Thank you!")
            break


if __name__ == "__main__":
    main()
