import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
# ensure repo root is on sys.path so local imports like `quantum_ml` resolve
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# load dataset and medians
csv = ROOT / 'classical_ml' / 'data' / 'house_price_top10.csv'
df = pd.read_csv(csv)
medians = df.median()

# user input
inp = {
    'square': 180,
    'price_per_sqm': 5.5,
    'communityaverage': 80000,
    'totalprice': 990000,
}
print('Input:', inp)

# load model features whitelist
feat_path = ROOT / 'classical_ml' / 'data' / 'model_features.json'
if not feat_path.exists():
    raise SystemExit('model_features.json not found; run smoke_run.py first')
FEATURES = json.load(open(feat_path))
print('Model FEATURES used:', FEATURES)

# --- ML prediction ---
model_dir = ROOT / 'classical_ml' / 'models'
candidates = ['rf_production_1766088474.pkl', 'rf_best.pkl', 'rf_retrained.pkl', 'rf_final_1766088474.pkl']
ml_model = None
ml_path = None
for c in candidates:
    p = model_dir / c
    if p.exists():
        try:
            ml_model = joblib.load(p)
            ml_path = p
            break
        except Exception as e:
            print('Could not load', p, e)

if ml_model is None:
    print('No ML model available; using fallback heuristic')
    def norm(v,c):
        mi = df[c].min(); ma = df[c].max();
        if ma<=mi: return 0.5
        return (v-mi)/(ma-mi)
    ml_prob = (0.1*norm(inp['square'],'square') + 0.5*norm(inp['price_per_sqm'],'price_per_sqm') + 0.2*norm(inp['communityaverage'],'communityaverage') + 0.2*norm(inp['totalprice'],'totalprice'))
else:
    print('Loaded ML model from', ml_path)
    # align input
    X = pd.DataFrame([inp])
    if hasattr(ml_model, 'feature_names_in_'):
        cols = list(ml_model.feature_names_in_)
        fill_vals = {c: float(medians[c]) if c in medians.index else 0.0 for c in cols}
        X = X.reindex(columns=cols).fillna(value=fill_vals)
    else:
        # pick FEATURES intersection
        cols = [c for c in FEATURES if c in df.columns]
        X = pd.DataFrame([[inp.get(c, float(medians.get(c,0))) for c in cols]], columns=cols)
    try:
        if hasattr(ml_model, 'predict_proba'):
            ml_prob = float(ml_model.predict_proba(X)[0][1])
        else:
            ml_prob = float(ml_model.predict(X)[0])
    except Exception as e:
        print('ML predict error:', e)
        ml_prob = None

ml_class = int(ml_prob >= 0.5) if ml_prob is not None else None
print(f"ML: class={ml_class}, prob_expensive={ml_prob}")

# --- Quantum prediction ---
try:
    from quantum_ml.quantum_preprocess import preprocess_data
    from quantum_ml.vqc_circuit import qnode
    # load shared split to fit scaler consistent with training
    split_dir = ROOT / 'classical_ml' / 'data'
    X_train = np.load(split_dir / 'X_train.npy')
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    # ensure features order matches FEATURES used by quantum preprocess
    # The saved X_train should already be in the correct FEATURE order
    scaler.fit(X_train)
    # build input vector in FEATURES order
    x_raw = np.array([[inp.get(f, 0) for f in FEATURES]])
    x_scaled = scaler.transform(x_raw)[0]
    params = np.load(ROOT / 'quantum_ml' / 'trained_params.npy', allow_pickle=True)
    q_out = qnode(params, x_scaled)
    q_prob = float((q_out + 1)/2)
    q_class = int(q_prob >= 0.5)
    print(f"VQC: class={q_class}, prob_expensive={q_prob}")
except Exception as e:
    print('Quantum evaluation failed:', e)
    q_prob = None
    q_class = None

# Summary
print('\nPer-input comparison:')
rows = [
    ['Classical ML', 'Expensive (1)' if ml_class==1 else ('Cheap (0)' if ml_class==0 else 'N/A'), f"{ml_prob:.2f}" if ml_prob is not None else 'N/A'],
    ['Quantum ML', 'Expensive (1)' if q_class==1 else ('Cheap (0)' if q_class==0 else 'N/A'), f"{q_prob:.2f}" if q_prob is not None else 'N/A']
]
print('Model\tPrediction\tProbability')
for r in rows:
    print(f"{r[0]}\t{r[1]}\t{r[2]}")

if ml_class is not None and q_class is not None:
    if ml_class == q_class:
        print('\n✅ Models agree on class')
    else:
        print('\n⚠️ Models disagree on class')
