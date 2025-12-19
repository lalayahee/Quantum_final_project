import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode

# Load trained parameters
params = np.load("trained_params.npy")

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
# Scaling
# -----------------------------
X_train, _, _, _ = preprocess_data()
scaler = MinMaxScaler(feature_range=(0, np.pi))
scaler.fit(X_train)
X_user_scaled = scaler.transform(X_user)[0]

# -----------------------------
# Quantum Output
# -----------------------------
q_out = qnode(params, X_user_scaled)
prob = (q_out + 1) / 2  # map [-1,1] -> [0,1]

# -----------------------------
# Bar chart values
# -----------------------------
bar_labels = ['Low (Cheap)', 'High (Expensive)']
bar_values = [1 - prob, prob]  # cheap vs expensive

# Determine the predicted class
prediction = 'Cheap' if prob < 0.5 else 'Expensive'
print(f"\nQuantum output ⟨Z⟩ = {q_out:.4f}, probability = {prob:.4f}")
print(f"Predicted: {prediction}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(5,5))
bars = plt.bar(bar_labels, bar_values, color=['skyblue', 'steelblue'])
plt.title('Quantum House Price Prediction')
plt.ylabel('Probability')

# Highlight the higher bar
for i, v in enumerate(bar_values):
    if v == max(bar_values):
        bars[i].set_color('orange')  # highlight the predicted class

plt.ylim(0,1)
plt.show()
