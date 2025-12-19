
# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix
# from quantum_preprocess import preprocess_data
# from vqc_circuit import qnode

# print("Loading data...")
# X_train, X_test, y_train, y_test = preprocess_data()

# print("Loading trained parameters...")
# params = np.load("trained_params.npy")

# def predict(x):
#     """
#     Quantum prediction:
#     expval >= 0 → 1 (high)
#     expval <  0 → 0 (cheap)
#     """
#     return 1 if qnode(params, x) >= 0 else 0

# y_pred = np.array([predict(x) for x in X_test])

# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))




import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from quantum_preprocess import preprocess_data
from vqc_circuit import qnode

print("Loading data...")
X_train, X_test, y_train, y_test = preprocess_data()

params = np.load("trained_params.npy")

def predict(x):
    # expval >= 0 → expensive (1)
    return 1 if qnode(params, x) >= 0 else 0

y_pred = np.array([predict(x) for x in X_test])

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
