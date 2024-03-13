import pennylane.numpy as np
import pennylane as qml


class Ansaetze:

    def circuit19(w: np.ndarray, n_qubits: int):
        """
        Creates a Circuit19 ansatz.

        Length of flattened vector must be n_qubits*3-1
        because for >1 qubits there are three gates

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
            n_qubits (int): number of qubits
        """
        if w is None:
            return n_qubits * 3 - 1

        w_idx = 0
        for q in range(n_qubits):
            qml.RX(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

        for q in range(n_qubits):
            if q > 0:
                qml.CRX(w[w_idx], wires=[(q + 1) % n_qubits, q])
                w_idx += 1

    def strongly_entangling(w: np.ndarray, n_qubits: int) -> None:
        """
        Creates a StronglyEntanglingLayers ansatz.

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits**2)
            n_qubits (int): number of qubits
        """
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
