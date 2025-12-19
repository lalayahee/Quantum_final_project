import numpy as np
from sklearn.preprocessing import MinMaxScaler
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode

params = np.load("trained_params.npy")

print("\n🔹 Quantum House Price Tester (4-Qubit Model) 🔹\n")

# -----------------------------
# Frontend Inputs
# -----------------------------
square = float(input("Home size (m²): "))
price_per_sqm = float(input("Price per sqm: "))
communityaverage = float(input("Community average price: "))
totalprice = float(input("Total price: "))

X_user = np.array([[
    square,
    price_per_sqm,
    communityaverage,
    totalprice
]])

# -----------------------------
# Scaling (same as training)
# -----------------------------
X_train, _, _, _ = preprocess_data()

scaler = MinMaxScaler(feature_range=(0, np.pi))
scaler.fit(X_train)

X_user_scaled = scaler.transform(X_user)[0]

# -----------------------------
# Prediction
# -----------------------------
q_out = qnode(params, X_user_scaled)
prob = (q_out + 1) / 2
prediction = 1 if prob >= 0.5 else 0

# -----------------------------
# Result
# -----------------------------
print("\n🧠 Quantum Output:", round(float(q_out), 4))
print("📊 Probability:", round(float(prob), 4))

if prediction == 0:
    print("🏷 Prediction: CHEAP HOUSE (0)")
else:
    print("🏷 Prediction: EXPENSIVE HOUSE (1)")
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode

# Load trained quantum parameters
params = np.load("trained_params.npy")

print("\n🔹 Quantum House Price Tester (4-Qubit Model) 🔹\n")

# -----------------------------
# User Inputs
# -----------------------------
square = float(input("Home size (m²): "))
price_per_sqm = float(input("Price per sqm: "))
communityaverage = float(input("Community average price: "))
totalprice = float(input("Total price: "))

X_user = np.array([[
    square,
    price_per_sqm,
    communityaverage,
    totalprice
]])

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

# -----------------------------
# Display results
# -----------------------------
print("\n🧠 Quantum Output (⟨Z⟩):", round(float(q_out), 4))

if prediction == 0:
    print("🏷 Prediction: CHEAP HOUSE (0)")
else:
    print("🏷 Prediction: EXPENSIVE HOUSE (1)")
