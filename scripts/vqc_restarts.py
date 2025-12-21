"""Run multiple VQC restarts, save per-run metrics and aggregated summary."""
import argparse
import json
import os
import time

# Ensure the project root is on sys.path so local package imports like `quantum_ml` work
import sys
from pathlib import Path
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

import numpy as np

from quantum_ml.train_vqc import train_vqc, n_layers

def run_restarts(n_restarts=3, epochs=30, max_train=500, batch_size=64, patience=5, lr=1e-2, seed=42):
    os.makedirs("quantum_ml/models", exist_ok=True)
    all_runs = []
    for i in range(n_restarts):
        s = seed + i
        print(f"\n=== Restart {i+1}/{n_restarts} (seed={s}) ===")
        res = train_vqc(epochs=epochs, lr=lr, n_layers=n_layers, max_train=max_train,
                        batch_size=batch_size, patience=patience, seed=s, return_params=True)
        run_summary = {
            "seed": s,
            "n_epochs": res.get("n_epochs"),
            "final_val_acc": res.get("final_val_acc"),
            "final_val_roc": res.get("final_val_roc"),
            "train_losses": res.get("train_losses"),
            "val_accs": res.get("val_accs"),
            "val_rocs": res.get("val_rocs"),
        }
        all_runs.append(run_summary)

    accs = [r["final_val_acc"] for r in all_runs if r["final_val_acc"] is not None]
    rocs = [r["final_val_roc"] for r in all_runs if r["final_val_roc"] is not None]
    summary = {
        "n_restarts": n_restarts,
        "acc_mean": float(np.mean(accs)) if accs else None,
        "acc_std": float(np.std(accs)) if accs else None,
        "roc_mean": float(np.mean(rocs)) if rocs else None,
        "roc_std": float(np.std(rocs)) if rocs else None,
        "runs": all_runs,
    }

    ts = int(time.time())
    out_path = f"quantum_ml/models/vqc_restarts_summary_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Saved restarts summary to:", out_path)
    print("Summary:", {k: summary[k] for k in ["acc_mean", "acc_std", "roc_mean", "roc_std"]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_restarts(n_restarts=args.n_restarts, epochs=args.epochs, max_train=args.max_train,
                 batch_size=args.batch_size, patience=args.patience, lr=args.lr, seed=args.seed)
