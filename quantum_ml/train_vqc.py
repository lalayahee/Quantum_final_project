# import pennylane as qml
# from pennylane import numpy as np
# from quantum_preprocess import preprocess_data
# from vqc_circuit import qnode, n_qubits

# n_layers = 2
# EPOCHS = 50
# LEARNING_RATE = 0.1

# def loss(params, X, y):
#     """MSE loss in quantum label space {-1, +1}."""
#     preds = np.array([qnode(params, x) for x in X])
#     return np.mean((preds - y) ** 2)

# if __name__ == "__main__":
#     print("Loading data...")
#     X_train, X_test, y_train, y_test = preprocess_data()

#     # Map labels: 0 → -1 (cheap), 1 → +1 (high)
#     y_train_q = 2 * y_train - 1

#     print("Initializing parameters...")
#     params = np.random.randn(n_layers, n_qubits, requires_grad=True)

#     optimizer = qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)

#     print("Starting training...\n")
#     for epoch in range(EPOCHS):
#         params = optimizer.step(lambda p: loss(p, X_train, y_train_q), params)

#         if epoch % 10 == 0:
#             l = loss(params, X_train, y_train_q)
#             print(f"Epoch {epoch:02d} | Loss: {l:.4f}")

#     np.save("trained_params.npy", np.array(params))
#     print("\nTraining completed.")
#     print("Saved trained parameters to trained_params.npy")



#  train vqc ( model )


import pennylane as qml
from pennylane import numpy as np
from quantum_ml.quantum_preprocess import preprocess_data
from quantum_ml.vqc_circuit import qnode, n_qubits

EPOCHS = 30
LEARNING_RATE = 1e-2
n_layers = 2
MAX_TRAIN = 500
BATCH_SIZE = 64
PATIENCE = 5

from sklearn.metrics import accuracy_score, roc_auc_score
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
import json, os, time


def _loss(params, X, y):
    preds = np.array([qnode(params, x) for x in X])
    return np.mean((preds - y) ** 2)


def train_vqc(epochs=EPOCHS, lr=LEARNING_RATE, n_layers=n_layers, max_train=MAX_TRAIN,
              batch_size=BATCH_SIZE, patience=PATIENCE, seed=None, return_params=True):
    """Train a single VQC run; returns dict with metrics and params."""
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = preprocess_data()

    # stratified subsample for faster iteration
    from sklearn.model_selection import StratifiedShuffleSplit
    if X_train.shape[0] > max_train:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_train, random_state=seed or 42)
        idx, _ = next(sss.split(X_train, y_train))
        X_train_sub = X_train[idx]
        y_train_sub = y_train[idx]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    y_train_q = 2 * y_train_sub - 1

    params = np.random.randn(n_layers, n_qubits, requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=lr)

    best_val = -1.0
    no_improve = 0
    train_losses = []
    val_accs = []
    val_rocs = []

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train_sub))
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            X_batch = X_train_sub[batch_idx]
            y_batch_q = (2 * y_train_sub[batch_idx] - 1)
            params = optimizer.step(lambda p: _loss(p, X_batch, y_batch_q), params)

        l = _loss(params, X_train_sub, y_train_q)
        train_losses.append(float(l))

        q_outs = np.array([qnode(params, x) for x in X_test])
        probs = (q_outs + 1) / 2.0
        y_pred = (probs >= 0.5).astype(int)
        val_acc = float(accuracy_score(y_test, y_pred))
        try:
            val_roc = float(roc_auc_score(y_test, probs))
        except Exception:
            val_roc = None

        val_accs.append(val_acc)
        val_rocs.append(val_roc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {l:.4f} | Val Acc: {val_acc:.4f} | Val ROC: {val_roc}")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping (no improvement).")
                break

    trained_params = np.array(params)
    result = {
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_rocs": val_rocs,
        "best_val_acc": best_val,
        "final_val_acc": val_accs[-1] if val_accs else None,
        "final_val_roc": val_rocs[-1] if val_rocs else None,
        "n_epochs": len(train_losses),
    }

    if return_params:
        result["params"] = trained_params

    os.makedirs("quantum_ml/models", exist_ok=True)
    ts = int(time.time())
    models_dir = "quantum_ml/models"
    np.save(f"{models_dir}/trained_params_{ts}.npy", trained_params)
    # Also save a canonical copy that the frontend expects at `quantum_ml/trained_params.npy`
    try:
        np.save("quantum_ml/trained_params.npy", trained_params)
    except Exception:
        pass

    with open(f"{models_dir}/vqc_metrics_{ts}.json", "w") as fh:
        json.dump({k: v if not hasattr(v, "dtype") else v.tolist() for k, v in result.items() if k != "params"}, fh, indent=2)

    if _HAS_MPL:
        plt.figure(figsize=(6,3))
        plt.plot(train_losses, label="train loss")
        plt.plot(val_accs, label="val acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"quantum_ml/models/vqc_training_{ts}.png", dpi=150)
        plt.close()
    else:
        print("matplotlib not available — skipping training plot save")

    print("Training completed. Artifacts saved in quantum_ml/models/")
    return result


if __name__ == "__main__":
    res = train_vqc()
    print("Run summary:", {"best_val_acc": res["best_val_acc"], "final_val_roc": res["final_val_roc"]})
