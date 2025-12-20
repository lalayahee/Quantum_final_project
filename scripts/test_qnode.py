import sys, os
sys.path.append(r'c:\Year 4\Quantum\Quantum_final_project')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# load params
params_path = r'c:\Year 4\Quantum\Quantum_final_project\quantum_ml\trained_params.npy'
params = np.load(params_path, allow_pickle=True)
# load qnode
from quantum_ml.vqc_circuit import qnode
# build scaler from raw csv
csv = r'c:\Year 4\Quantum\Quantum_final_project\classical_ml\data\house_price_top10.csv'
import pandas as pd
(df) = pd.read_csv(csv)
# Prefer model feature whitelist saved by the classical pipeline (if present)
from pathlib import Path
import json
model_feat = Path(__file__).resolve().parents[1] / 'classical_ml' / 'data' / 'model_features.json'
if model_feat.exists():
    FEATURES = json.load(open(model_feat))
else:
    FEATURES = ["square","communityaverage","renovationcondition","followers"]

# build X from CSV using the selected features
X_raw = df[FEATURES].dropna().values
scaler = MinMaxScaler(feature_range=(0, np.pi))
scaler.fit(X_raw)
# transform a sample taken from the data
x = X_raw[0:1]
xt = scaler.transform(x)[0]
print('Using FEATURES:', FEATURES)
print('X_scaled sample:', xt)
# call qnode
try:
    q_out = qnode(params, xt)
    print('q_out:', q_out, 'prob:', (q_out+1)/2)
except Exception as e:
    print('qnode error:', type(e), e)
