import json
import os
from pathlib import Path
# Ensure the project root is on sys.path so local package imports like `quantum_ml` work
import sys
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

root = Path(__file__).resolve().parents[1]
csv_path = root / 'classical_ml' / 'data' / 'house_price_top10.csv'
if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

# ensure target exists
if 'target' not in df.columns:
    target_col = 'totalprice' if 'totalprice' in df.columns else ('price' if 'price' in df.columns else None)
    if target_col is None:
        raise SystemExit('No price column to derive target from')
    df['target'] = (df[target_col] >= df[target_col].median()).astype(int)

# choose candidate features excluding leaks and geo
leak_cols = ['price','totalprice','price_per_sqm','log_price']
geo_cols = ['lng','lat','longitude','latitude']
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cand = [c for c in num_cols if c not in leak_cols + geo_cols + ['target']]

print('Candidate numeric features:', cand[:20])

# fallback defaults
fallback = ["square", "communityaverage", "renovationcondition", "followers"]
features = [f for f in cand if f in df.columns]
if len(features) < 4:
    features = [f for f in fallback if f in df.columns]

if len(features) == 0:
    raise SystemExit('No valid features found for training')

# if more than 4, pick top 4 by RF importance on small subsample
if len(features) > 4:
    sample = df.dropna(subset=features + ['target']).sample(min(5000, len(df)), random_state=42)
    X_s = sample[features]
    y_s = sample['target']
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_s, y_s)
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    features = imp.head(4).index.tolist()

print('Selected features:', features)

# create train/test split
X = df[features].fillna(0)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

out_dir = root / 'classical_ml' / 'data'
out_dir.mkdir(parents=True, exist_ok=True)
# save features and split
with open(out_dir / 'model_features.json', 'w') as f:
    json.dump(features, f)
np.save(out_dir / 'X_train.npy', X_train.to_numpy())
np.save(out_dir / 'X_test.npy', X_test.to_numpy())
np.save(out_dir / 'y_train.npy', y_train.to_numpy())
np.save(out_dir / 'y_test.npy', y_test.to_numpy())
print('Saved model_features.json and X/y numpy splits to', out_dir)

# quick RF training smoke test on the same features
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
try:
    y_proba = rf.predict_proba(X_test)[:,1]
except Exception:
    y_proba = None
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
print(f'RF smoke: accuracy={acc:.4f}, roc_auc={auc}')

# Try loading quantum preprocess to confirm it picks up the split
try:
    from quantum_ml.quantum_preprocess import preprocess_data
    Xtr, Xte, ytr, yte = preprocess_data()
    print('Quantum preprocess returned shapes:', Xtr.shape, Xte.shape, ytr.shape, yte.shape)
except Exception as e:
    print('Quantum preprocess error:', e)

print('Smoke run completed.')
