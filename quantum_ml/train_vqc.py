
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






import pennylane as qml
from pennylane import numpy as np
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode, n_qubits

EPOCHS = 50
LEARNING_RATE = 0.1
n_layers = 2

def loss(params, X, y):
    preds = np.array([qnode(params, x) for x in X])
    return np.mean((preds - y) ** 2)

print("Loading data...")
X_train, X_test, y_train, y_test = preprocess_data()

# Map labels: 0 → -1 (cheap), 1 → +1 (expensive)
y_train_q = 2 * y_train - 1

params = np.random.randn(n_layers, n_qubits, requires_grad=True)
optimizer = qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)

print("Training started...\n")
for epoch in range(EPOCHS):
    params = optimizer.step(lambda p: loss(p, X_train, y_train_q), params)

    if epoch % 10 == 0:
        l = loss(params, X_train, y_train_q)
        print(f"Epoch {epoch:02d} | Loss: {l:.4f}")

np.save("trained_params.npy", np.array(params))
print("\nTraining completed. Parameters saved.")
