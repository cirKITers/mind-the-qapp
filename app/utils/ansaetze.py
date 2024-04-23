import pennylane.numpy as np
import pennylane as qml


class Ansaetze:
    def get_available():
        return [Ansaetze.no_ansatz, Ansaetze.circuit19, Ansaetze.strongly_entangling]

    @staticmethod
    def no_ansatz(w: np.ndarray, n_qubits: int):
        if w is None:
            return 1
        pass

    @staticmethod
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
            if n_qubits > 1:
                return n_qubits * 3
            else:
                return 2

        w_idx = 0
        for q in range(n_qubits):
            qml.RX(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

        if n_qubits > 1:
            for q in range(n_qubits):
                qml.CRX(w[w_idx], wires=[n_qubits - q - 1, (n_qubits - q) % n_qubits])
                w_idx += 1

    @staticmethod
    def strongly_entangling(w: np.ndarray, n_qubits: int) -> None:
        """
        Creates a StronglyEntanglingLayers ansatz.

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
            n_qubits (int): number of qubits
        """
        if w is None:
            return n_qubits * 3

        qml.StronglyEntanglingLayers(w.reshape(-1, n_qubits, 3), wires=range(n_qubits))
