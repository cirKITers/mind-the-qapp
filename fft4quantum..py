import pennylane as qml

from functools import partial

import time

from pennylane import numpy as np
from pennylane.fourier import coefficients
from pennylane.fourier.visualize import *

rng = np.random.default_rng(1111)

n_qubits = 2
n_layers = 4

weights = rng.random(size=(n_layers, n_qubits, 2))

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def circuit_with_weights(w, x):
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RX(x, wires=q)

            qml.RY(w[l, q, 0], wires=q)
            qml.RZ(w[l, q, 1], wires=q)

        # for q in range(n_qubits):
        #     qml.CRX(w[l, q, 2], wires=[q, (q + 1) % n_qubits])

    return qml.expval(qml.PauliZ(0))


x = 0.1

start = time.time()

coeffs = coefficients(partial(circuit_with_weights, weights), 1, n_qubits * n_layers)

omega = np.arange(0, len(coeffs))


f_x_fft = 0
for i, coeff in enumerate(coeffs):
    # print(f"Coefficient c_-{omega[i]}=c_+{omega[i]} = {coeff}")
    f_x_fft += coeff * np.exp(-1j * x * omega[i])

stop = time.time()

print(f"Calculation using coefficients took {stop-start}")
print(f"Result: {np.real(f_x_fft)}")

start = time.time()

res = circuit_with_weights(weights, x)

stop = time.time()

print(f"Calculation using pennylane took {stop-start}")
print(f"Result: {res}")
