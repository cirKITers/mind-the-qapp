import pennylane.numpy as np
import pennylane as qml


class Ansaetze:
    def get_available():
        return [
            Ansaetze.no_ansatz,
            Ansaetze.circuit01,
            Ansaetze.circuit19,
            Ansaetze.no_entangling,
            Ansaetze.strongly_entangling,
            Ansaetze.idle,
        ]

    @staticmethod
    def no_ansatz(w: np.ndarray, n_qubits: int):
        if w is None:
            return 1
        pass

    @staticmethod
    def circuit01(w: np.ndarray, n_qubits: int):
        """
        Creates a Circuit1 ansatz.

        Length of flattened vector must be n_qubits*2

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*2)
            n_qubits (int): number of qubits
        """
        if w is None:
            return n_qubits * 2

        w_idx = 0
        for q in range(n_qubits):
            qml.RX(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

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
            return n_qubits * 6

        w_idx = 0
        for q in range(n_qubits):
            qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
            w_idx += 3

        if n_qubits > 1:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        for q in range(n_qubits):
            qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
            w_idx += 3

        if n_qubits > 1:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + n_qubits // 2) % n_qubits])

        # qml.StronglyEntanglingLayers(w.reshape(-1, n_qubits, 3), wires=range(n_qubits))

    @staticmethod
    def no_entangling(w: np.ndarray, n_qubits: int):
        """
        Creates a circuit without entangling, but with U3 gates on all qubits

        Length of flattened vector must be n_qubits*3

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3)
            n_qubits (int): number of qubits
        """
        if w is None:
            return n_qubits * 3

        w_idx = 0
        for q in range(n_qubits):
            qml.Rot(w[w_idx], w[w_idx + 1], w[w_idx + 2], wires=q)
            w_idx += 3

    @staticmethod
    def idle(w: np.ndarray, n_qubits: int) -> None:
        """
        Creates an idle circuit, which does nothing

        Args:
            w (np.ndarray): weights (are not used)
            n_qubits (int): number of qubits
        """
        if w is None:
            return n_qubits

        # FIXME: for batching purpuses, instead of just applying identity
        # gates, we rotate by zeros
        w = np.zeros_like(w)

        for q in range(n_qubits):
            qml.RX(w[0], wires=q)
