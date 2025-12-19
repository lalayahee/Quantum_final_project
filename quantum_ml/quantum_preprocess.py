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

    # ✅ SELECT 4 FEATURES
    FEATURES = ["square", "price_per_sqm", "communityaverage", "totalprice"]
    X = df[FEATURES].values
    y = df["target"].values  # 0 = cheap, 1 = expensive

    # Scale features to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

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
