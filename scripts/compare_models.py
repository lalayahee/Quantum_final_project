import json, os
from pathlib import Path
# Ensure the project root is on sys.path so local package imports like `quantum_ml` work
import sys
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
split_dir = ROOT / 'classical_ml' / 'data'
model_dir = ROOT / 'classical_ml' / 'models'
model_dir.mkdir(parents=True, exist_ok=True)

# Load shared split and feature list
X_train = np.load(split_dir / 'X_train.npy')
X_test = np.load(split_dir / 'X_test.npy')
y_train = np.load(split_dir / 'y_train.npy')
y_test = np.load(split_dir / 'y_test.npy')
with open(split_dir / 'model_features.json') as f:
    features = json.load(f)

print('Loaded split:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print('Features:', features)

results = {}
# 1) Train a quick RandomForest on the raw features (classical baseline)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]

results['rf'] = {
    'accuracy': float(accuracy_score(y_test, rf_pred)),
    'roc_auc': float(roc_auc_score(y_test, rf_proba)),
    'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist(),
    'classification_report': classification_report(y_test, rf_pred, output_dict=True)
}
print('RF done')

# 2) Evaluate VQC using trained params and scaled test split
try:
    import pennylane as qml
    from pennylane import numpy as np2
    from quantum_ml.vqc_circuit import qnode
    from quantum_ml.quantum_preprocess import preprocess_data

    # preprocess_data will load the shared split and scale to [0, pi]
    X_tr_scaled, X_te_scaled, y_tr_scaled, y_te_scaled = preprocess_data()
    # Use X_te_scaled and y_te_scaled
    params = np.load(ROOT / 'quantum_ml' / 'trained_params.npy', allow_pickle=True)

    q_outs = np2.array([qnode(params, x) for x in X_te_scaled])
    q_probs = ((q_outs + 1) / 2).astype(float)
    q_preds = (q_probs >= 0.5).astype(int)

    results['vqc'] = {
        'accuracy': float(accuracy_score(y_te_scaled, q_preds)),
        'roc_auc': float(roc_auc_score(y_te_scaled, q_probs)),
        'confusion_matrix': confusion_matrix(y_te_scaled, q_preds).tolist(),
        'classification_report': classification_report(y_te_scaled, q_preds, output_dict=True)
    }
    print('VQC done')
except Exception as e:
    print('VQC evaluation failed:', e)
    results['vqc'] = {'error': str(e)}

# Save comparison
out_path = model_dir / 'comparison_vqc_rf.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

# Pretty print summary table
print('\nComparison summary:')
print(f"RF:  acc={results['rf']['accuracy']:.4f}, auc={results['rf']['roc_auc']:.4f}")
if 'error' in results['vqc']:
    print('VQC: error —', results['vqc']['error'])
else:
    print(f"VQC: acc={results['vqc']['accuracy']:.4f}, auc={results['vqc']['roc_auc']:.4f}")

print('\nSaved comparison to', out_path)

# --- Create a saved comparison bar chart (Accuracy & ROC AUC) for quick visual inspection ---
try:
    import matplotlib.pyplot as plt
    plot_dir = model_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = ['RF', 'VQC']
    acc_rf = results['rf']['accuracy']
    auc_rf = results['rf']['roc_auc']
    if 'error' in results['vqc']:
        acc_vqc = 0.0
        auc_vqc = 0.0
    else:
        acc_vqc = results['vqc']['accuracy']
        auc_vqc = results['vqc']['roc_auc']

    acc_vals = [acc_rf, acc_vqc]
    auc_vals = [auc_rf, auc_vqc]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='#1f77b4')
    ax.bar(x + width/2, auc_vals, width, label='ROC AUC', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Metric')
    ax.legend()
    ax.set_title('RF vs VQC: Accuracy & ROC AUC')
    out_plot = plot_dir / 'comparison_bars.png'
    fig.tight_layout()
    fig.savefig(out_plot)
    print('Saved comparison plot to', out_plot)
except Exception as e:
    print('Could not save comparison plot:', e)
