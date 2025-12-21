import numpy as np
from sklearn.preprocessing import MinMaxScaler
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode
from pathlib import Path
import json

# Load trained quantum parameters
params = np.load("trained_params.npy")

print("\n🔹 Quantum House Price Tester (4-Qubit Model) 🔹\n")

# -----------------------------
# User Inputs (interface unchanged)
# -----------------------------
square = float(input("Home size (m²): "))
price_per_sqm = float(input("Price per sqm: "))
communityaverage = float(input("Community average price: "))
totalprice = float(input("Total price: "))

# Keep the full payload (frontend interface unchanged)
user_payload = {
    "square": square,
    "price_per_sqm": price_per_sqm,
    "communityaverage": communityaverage,
    "totalprice": totalprice,
}

# -----------------------------
# Determine model features (load whitelist if available)
# -----------------------------
feat_path = Path(__file__).resolve().parents[1] / "classical_ml" / "data" / "model_features.json"
if feat_path.exists():
    with open(feat_path, 'r') as f:
        FEATURES = json.load(f)
else:
    FEATURES = ["square", "communityaverage", "renovationcondition", "followers"]

# Build model input in the expected feature order (missing keys -> 0)
X_user = np.array([[user_payload.get(f, 0) for f in FEATURES]])

# -----------------------------
# Scaling (same as training)
# -----------------------------
X_train, _, _, _ = preprocess_data()

scaler = MinMaxScaler(feature_range=(0, np.pi))
scaler.fit(X_train)  # fit only on training features

X_user_scaled = scaler.transform(X_user)[0]

# -----------------------------
# Quantum Prediction
# -----------------------------
q_out = qnode(params, X_user_scaled)

# Decision rule consistent with training: ⟨Z⟩ >= 0 → expensive
prediction = 1 if q_out >= 0 else 0
prob = (q_out + 1) / 2

# -----------------------------
# Display results
# -----------------------------
print("\n🧠 Quantum Output (⟨Z⟩):", round(float(q_out), 4))
print("📊 Probability:", round(float(prob), 4))

if prediction == 0:
    print("🏷 Prediction: CHEAP HOUSE (0)")
else:
    print("🏷 Prediction: EXPENSIVE HOUSE (1)")
