from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, pandas as pd, json, os
from pathlib import Path

app = FastAPI()

class Item(BaseModel):
    data: dict

# robust base dir (works both as module and in notebook)
base_dir = Path(__file__).resolve().parents[0]
MODEL_PATH = os.path.join(base_dir, "models")
# pick latest production model if present
model_file = None
if os.path.exists(MODEL_PATH):
    candidates = [p for p in os.listdir(MODEL_PATH) if p.endswith('.pkl')]
    if candidates:
        # choose newest
        candidates.sort()
        model_file = os.path.join(MODEL_PATH, candidates[-1])

MODEL_PATH = model_file if model_file is not None else os.path.join(base_dir, "models", "rf_production_1766081555.pkl")

# Load feature whitelist if available
feature_file = os.path.join(base_dir, "data", "model_features.json")
if os.path.exists(feature_file):
    with open(feature_file, 'r') as f:
        MODEL_FEATURES = json.load(f)
else:
    # fallback list (must match quantum defaults)
    MODEL_FEATURES = ["square", "communityaverage", "renovationcondition", "followers"]

# load model safely and warn if unavailable
_model = None
try:
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: could not find a model at {MODEL_PATH}")
except Exception as e:
    print(f"Warning: could not load model at {MODEL_PATH}: {e}")

@app.post("/predict")
def predict(item: Item):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    df = pd.DataFrame([item.data])

    # Keep full input payload in response but construct a model-only dataframe by whitelisting features
    df_model = df.reindex(columns=MODEL_FEATURES).fillna(0)

    try:
        probs = _model.predict_proba(df_model)[:, 1].tolist()
    except Exception:
        probs = None
    preds = _model.predict(df_model).tolist()
    return {"input": item.data, "predictions": preds, "probabilities": probs}