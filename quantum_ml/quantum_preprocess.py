# --- 2 qbits ---
# import sys
# from pathlib import Path
# import argparse

# try:
#     import pandas as pd
#     import numpy as np
#     from sklearn.preprocessing import MinMaxScaler
#     from sklearn.model_selection import train_test_split
# except ModuleNotFoundError as e:
#     missing = e.name
#     print(f"Missing Python package: {missing}.\nInstall requirements with: pip install -r ../requirement.txt")
#     raise


# def preprocess_data(csv_path: str | Path | None = None):
#     """Load CSV, scale features, sample dataset, and return train/test split."""

#     # Default path resolution
#     if csv_path is None:
#         repo_root = Path(__file__).resolve().parents[1]
#         candidates = [
#             repo_root / "classical_ml" / "data" / "house_price_top10.csv",
#             repo_root / "data" / "house_price_top10.csv",
#             Path.cwd() / "house_price_top10.csv",
#         ]
#         found = None
#         for c in candidates:
#             if c.exists():
#                 found = c
#                 break
#         if found is None:
#             msg = (
#                 "CSV file not found. Checked these locations:\n"
#                 + "\n".join(str(p) for p in candidates)
#                 + "\nPlease provide the correct path with --csv or place the file in one of the above locations."
#             )
#             raise FileNotFoundError(msg)
#         csv_path = found
#     csv_path = Path(csv_path)

#     # Load data
#     df = pd.read_csv(csv_path)
#     X = df[["square", "price_per_sqm"]].values
#     y = df["target"].values

#     # Scale features to [0, pi] for quantum circuit
#     scaler = MinMaxScaler(feature_range=(0, np.pi))
#     X_scaled = scaler.fit_transform(X)

#     # Reduce dataset size for quantum training
#     MAX_SAMPLES = 5000
#     if len(X_scaled) > MAX_SAMPLES:
#         X_small, _, y_small, _ = train_test_split(
#             X_scaled,
#             y,
#             train_size=MAX_SAMPLES,
#             stratify=y,
#             random_state=42
#         )
#     else:
#         X_small, y_small = X_scaled, y

#     print("Original dataset size:", len(df))
#     print("Sampled dataset size:", len(X_small))

#     return train_test_split(
#         X_small,
#         y_small,
#         test_size=0.3,
#         random_state=42
#     )


# def _parse_args():
#     p = argparse.ArgumentParser(description="Preprocess house price CSV for quantum ML")
#     p.add_argument("--csv", help="Path to CSV file (default: ../data/house_price_top10.csv)")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = _parse_args()
#     try:
#         X_train, X_test, y_train, y_test = preprocess_data(args.csv)
#     except FileNotFoundError as e:
#         print(e)
#         sys.exit(2)

#     print("Preprocessing OK")
#     print("X_train shape:", getattr(X_train, 'shape', None))
#     print("y_train shape:", getattr(y_train, 'shape', None))

#     # show a small sample
#     try:
#         import pandas as _pd
#         repo_root = Path(__file__).resolve().parents[1]
#         sample_csv = repo_root / "data" / "house_price_top10.csv"
#         if Path(args.csv or sample_csv).exists():
#             df_sample = _pd.read_csv(args.csv or sample_csv)
#             print(df_sample.head())
#     except Exception:
#         pass


import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_data(csv_path: str | Path | None = None):
    """Load CSV, scale features, sample dataset, and return train/test split."""

    if csv_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            repo_root / "classical_ml" / "data" / "house_price_top10.csv",
            repo_root / "data" / "house_price_top10.csv",
            Path.cwd() / "house_price_top10.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError("CSV file not found.")

    df = pd.read_csv(csv_path)

    # ✅ SELECT features for MODEL (exclude price-derived columns to prevent label leakage)
    # Keep price-derived fields available in the dataframe for the frontend, but DO NOT use them for training.
    # Default non-leaky features (change if you intentionally want others)
    DEFAULT_FEATURES = ["square", "communityaverage", "renovationcondition", "followers"]
    FEATURES = DEFAULT_FEATURES

    # If the classical pipeline saved a `model_features.json` whitelist, prefer it (but exclude price-derived cols)
    model_feat_path = Path(__file__).resolve().parents[1] / "classical_ml" / "data" / "model_features.json"
    if model_feat_path.exists():
        try:
            import json as _json
            mf = _json.load(open(model_feat_path))
            leak_cols = ['price','totalprice','price_per_sqm','log_price']
            mf = [f for f in mf if f in df.columns and f not in leak_cols]
            if mf:
                FEATURES = mf
                print("Using model features from", model_feat_path)
        except Exception:
            pass

    X = df[FEATURES].values
    y = df["target"].values  # 0 = cheap, 1 = expensive

    # Scale features to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # If a shared split (saved by the classical ML preprocessing) exists, load it so both pipelines use the same data.
    split_dir = Path(__file__).resolve().parents[1] / "classical_ml" / "data"
    x_train_npy = split_dir / "X_train.npy"
    y_train_npy = split_dir / "y_train.npy"
    x_test_npy = split_dir / "X_test.npy"
    y_test_npy = split_dir / "y_test.npy"
    if x_train_npy.exists() and y_train_npy.exists() and x_test_npy.exists() and y_test_npy.exists():
        X_train = np.load(x_train_npy)
        X_test = np.load(x_test_npy)
        y_train = np.load(y_train_npy)
        y_test = np.load(y_test_npy)
        print("Loaded shared train/test split from:", split_dir)
        # For quantum training we want a manageable sample size. If the classical split is large,
        # subsample the training set (stratified) to MAX_Q_SAMPLES to keep circuit count reasonable.
        MAX_Q_SAMPLES = 1000
        if X_train.shape[0] > MAX_Q_SAMPLES:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_Q_SAMPLES, random_state=42)
            sub_idx, _ = next(sss.split(X_train, y_train))
            X_train = X_train[sub_idx]
            y_train = y_train[sub_idx]
            print(f"Subsampled training set to {MAX_Q_SAMPLES} examples for quantum training")
        # fit scaler on training split and transform both train and test
        try:
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            print("Scaled shared split features to [0, pi] for quantum training")
        except Exception:
            pass
        return X_train, X_test, y_train, y_test

    MAX_SAMPLES = 5000
    if len(X_scaled) > MAX_SAMPLES:
        X_small, _, y_small, _ = train_test_split(
            X_scaled, y, train_size=MAX_SAMPLES, stratify=y, random_state=42
        )
    else:
        X_small, y_small = X_scaled, y

    print("Original dataset size:", len(df))
    print("Sampled dataset size:", len(X_small))

    return train_test_split(
        X_small, y_small, test_size=0.3, stratify=y_small, random_state=42
    )
