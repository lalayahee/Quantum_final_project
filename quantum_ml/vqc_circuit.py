
# import pennylane as qml
# from pennylane import numpy as np

# n_qubits = 2
# n_layers = 2 # number of variational layers
# dev = qml.device("default.qubit", wires=n_qubits)


# # def vqc(params, x):
# #     # Encode input
# #     for i in range(n_qubits):
# #         qml.RY(x[i], wires=i) # Encode features as rotation angles

# #     # Variational parameters
# #     # for i in range(n_qubits):
# #     #     qml.RY(params[i], wires=i)

# #     for layer in range(n_layers):
# #         for i in range(n_qubits):
# #             qml.RY(params[layer, i], wires=i)

# #     # Entanglement
# #     qml.CNOT(wires=[0, 1])

# #     return qml.expval(qml.PauliZ(0))
# def vqc(params, x):
#     for layer in range(n_layers):
#         # data re-uploading
#         for i in range(n_qubits):
#             qml.RY(x[i], wires=i)
#             qml.RY(params[layer, i], wires=i)

#         # entanglement per layer
#         for i in range(n_qubits - 1):
#             qml.CNOT(wires=[i, i + 1])

#     return qml.expval(qml.PauliZ(0))



# # QNode wraps the quantum function
# qnode = qml.QNode(vqc, dev)



#  jea curcuit bos yg ( cnot ,......)

import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

def vqc(params, x):
    """4-qubit Variational Quantum Classifier"""
    for layer in range(n_layers):
        # Data re-uploading + variational rotations
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
            qml.RY(params[layer, i], wires=i)

        # Linear entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))

qnode = qml.QNode(vqc, dev)
