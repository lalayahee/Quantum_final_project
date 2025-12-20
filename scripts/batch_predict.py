import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load data medians
csv = os.path.join(ROOT, 'classical_ml', 'data', 'house_price_top10.csv')
df = pd.read_csv(csv)
medians = df[['square','price_per_sqm','communityaverage','totalprice']].median()

# helper norm used by fallback
def compute_ml_fallback_prob_local(inp, df_local):
    FEATURES = ['square','price_per_sqm','communityaverage','totalprice']
    if df_local is None or df_local.empty:
        mins = {'square':10.0,'price_per_sqm':1.0,'communityaverage':1.0,'totalprice':10.0}
        maxs = {'square':500.0,'price_per_sqm':12.0,'communityaverage':200000.0,'totalprice':5000.0}
    else:
        mins = df_local[FEATURES].min().to_dict()
        maxs = df_local[FEATURES].max().to_dict()
    def norm(v,c):
        mi = mins.get(c,0.0); ma = maxs.get(c,1.0)
        if ma<=mi: return 0.5
        return (v-mi)/(ma-mi)
    n = [np.clip(norm(inp[c],c),0,1) for c in FEATURES]
    w = np.array([0.1,0.5,0.2,0.2])
    score = float(np.dot(w,n))
    prob = 1.0/(1.0+float(np.exp(-4.0*(score-0.5))))
    return prob, score

# load ML model
model_dir = os.path.join(ROOT,'classical_ml','models')
candidates = ['rf_production_1766088474.pkl','rf_best.pkl','best.pkl']
ml_model = None; ml_path=None
for c in candidates:
    p = os.path.join(model_dir,c)
    if os.path.exists(p):
        try:
            ml_model = joblib.load(p)
            ml_path = p
            break
        except Exception as e:
            print('Could not load',p,e)

# load quantum params and scaler
params_path = os.path.join(ROOT,'quantum_ml','trained_params.npy')
PARAMS = None
try:
    from quantum_ml.vqc_circuit import qnode
    PARAMS = np.load(params_path,allow_pickle=True)
except Exception as e:
    qnode = None

# scaler based on X_train
scaler = None
try:
    X_train = np.load(os.path.join(ROOT,'classical_ml','data','X_train.npy'))
    scaler = MinMaxScaler(feature_range=(0,np.pi))
    scaler.fit(X_train)
except Exception:
    scaler=None

inputs = [
    {'name':'User example','square':180,'price_per_sqm':5.5,'communityaverage':80000,'totalprice':990000},
    {'name':'Small cheap','square':50,'price_per_sqm':3.0,'communityaverage':20000,'totalprice':150000},
    {'name':'Median','square':float(medians['square']),'price_per_sqm':float(medians['price_per_sqm']),'communityaverage':float(medians['communityaverage']),'totalprice':float(medians['totalprice'])},
    {'name':'Large expensive','square':400,'price_per_sqm':10.0,'communityaverage':300000,'totalprice':4000000}
]

print('ML model loaded from:', ml_path)
print('Quantum params loaded:', bool(PARAMS))
print('Scaler loaded:', bool(scaler))
print('\n')

for inp in inputs:
    print('---')
    print('Input:', inp['name'])
    print({k: inp[k] for k in ['square','price_per_sqm','communityaverage','totalprice']})
    # ML
    if ml_model is not None:
        X = pd.DataFrame([{'square':inp['square'],'price_per_sqm':inp['price_per_sqm'],'communityaverage':inp['communityaverage'],'totalprice':inp['totalprice']}])
        try:
            if hasattr(ml_model,'feature_names_in_'):
                cols = list(ml_model.feature_names_in_)
                fill_vals = {c: float(medians[c]) if c in medians.index else 0.0 for c in cols}
                X = X.reindex(columns=cols).fillna(value=fill_vals)
        except Exception:
            pass
        try:
            if hasattr(ml_model,'predict_proba'):
                probs = ml_model.predict_proba(X)[0]
                classes = list(getattr(ml_model,'classes_',[]))
                pos_idx = classes.index(1) if 1 in classes else (-1)
                ml_prob_raw = float(probs[pos_idx])
            else:
                ml_prob_raw = float(ml_model.predict(X)[0])
            print('ML raw prob (High class):', ml_prob_raw)
            print('ML classes:', getattr(ml_model,'classes_',None))
            print('ML probs (full):', probs if 'probs' in locals() else None)
        except Exception as e:
            print('ML predict error:', e)
            ml_prob_raw = None
    else:
        ml_prob_raw, score = compute_ml_fallback_prob_local(inp, df)
        print('ML fallback score:', score)
        print('ML fallback prob:', ml_prob_raw)

    # Quantum
    q_prob = None
    try:
        if PARAMS is not None and scaler is not None and qnode is not None:
            X_raw = np.array([[inp['square'], inp['price_per_sqm'], inp['communityaverage'], inp['totalprice']]])
            X_scaled = scaler.transform(X_raw)[0]
            q_out = qnode(PARAMS, X_scaled)
            q_prob = float((q_out+1)/2)
            print('Quantum prob (High):', q_prob)
    except Exception as e:
        print('Quantum error:', e)

    # recommended toggles
    print('\nRecommended UI settings:')
    print('- Calibration: ON (if available)')
    print('- Smoothing shrink factor: try 0.8 if ML seems overconfident')
    print('\n')

print('Done')
