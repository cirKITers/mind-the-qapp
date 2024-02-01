import pennylane as qml
import numpy as np
from typing import Optional
import tensorflow as tf

def ising_hamiltonian(J: np.ndarray, h: Optional[np.ndarray] = None) \
        -> qml.Hamiltonian:
    """
    Creates pennylane observable based on the Ising matrix and linear terms

    :param J: np.ndarray: Ising matrix
    :param h: np.ndarray: linear Ising terms

    :return: qml.Hamiltonian: Observable
    """

    # Initialize empty Hamiltonian
    coefficients, op_list = [], []

    # Linear Terms
    if h is not None:
        for i, angle in enumerate(h):
            if angle > 0:
                coefficients.append(angle)
                op_list.append(qml.PauliZ(i))

    # Quadratic Terms (Assuming a Ising matrix with zero diagonal elements)
    n = J.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if J[i][j]> 0 or J[j][i] > 0:
                coefficients.append(J[i][j] + J[j][i])
                op_list.append(qml.PauliZ([i]) @ qml.PauliZ(j))

    hamiltonian = qml.Hamiltonian(coefficients, op_list, simplify=True)

    return hamiltonian


def circ_19(params: tf.Variable, observable: qml.Hamiltonian) -> qml.measurements.ExpectationMP:
    """
    Circuit 19 from https://arxiv.org/abs/1905.10876

    :param params: tf.Variable: Parameters for rotation gates of shape
                (n_layers, 3 * n_qubits) or (n_layers, 3, n_qubits)
    :param observable: qml.Hamiltonian: Target Hamiltonian

    :return qml.measurements.ExpectationMP: Expectation for target Hamiltonian
    """
    assert len(params.shape) == 2  and params.shape[1] % 3 == 0 or \
            len(params.shape) == 3 and params.shape[1] == 3, \
    "Parameters for circuit 19 should either be of shape" \
    "(n_layers, 3 * n_qubits) or (n_layers, 3, n_qubits)"

    n_layers = params.shape[0]
    n_qubits = params.shape[1] // 3 if len(params.shape) == 2 else params.shape[2]

    qml.BasisState(np.zeros(n_qubits), range(n_qubits))

    for l in range(n_layers):
        p = tf.split(params[l], 3) if len(params.shape) == 2 else params[l]

        for i in range(n_qubits):
            qml.RX(p[0][i], i)
            qml.RZ(p[1][i], i)

        for i in range(n_qubits-1, -1, -1):
            qml.CRX(p[2][i], [i, (i+1) % n_qubits])

    return qml.expval(observable)

