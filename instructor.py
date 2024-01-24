from typing import Any

import pennylane as qml
import pennylane.numpy as np


class Model:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.mixed", wires=n_qubits)

        self.circuit = qml.QNode(self._circuit, self.dev)

    def pqc(self, w: np.ndarray):
        """
        Creates a Circuit19 ansatz.

        Length of flattened vector must be n_qubits*3-1
        because for >1 qubits there are three gates

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
        """

        w_idx = 0
        for q in range(self.n_qubits):
            qml.RX(w[w_idx], wires=q)
            w_idx += 1
            qml.RZ(w[w_idx], wires=q)
            w_idx += 1

            if q > 0:
                qml.CRX(w[w_idx], wires=[q, (q + 1) % self.n_qubits])
                w_idx += 1

    def iec(self, x: np.ndarray):
        """
        Creates an AngleEncoding using RY gates

        Args:
            x (np.ndarray): length of vector must be 1
        """

        for q in range(self.n_qubits):
            qml.RY(x, wires=q)

    def _circuit(
        self, w: np.ndarray, x: np.ndarray, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0
    ):
        """
        Creates a circuit with noise.
        This involves, Amplitude Damping, Phase Damping and Depolarization.
        The Circuit consists of a PQC and IEC in each layer.

        Args:
            w (np.ndarray): weight vector of size n_layers*(n_qubits*3-1)
            x (np.ndarray): input vector of size 1
            ad (float, optional): Amplitude Damping. Defaults to 0.0.
            pd (float, optional): Phase Damping. Defaults to 0.0.
            dp (float, optional): Depolarization. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        for l in range(self.n_layers):
            self.pqc(w[l])
            self.iec(x)

            for q in range(self.n_qubits):
                qml.BitFlip(bf, wires=q)
                qml.PhaseFlip(pf, wires=q)
                qml.AmplitudeDamping(ad, wires=q)
                qml.PhaseDamping(pd, wires=q)
                qml.DepolarizingChannel(dp, wires=q)

        return qml.expval(qml.PauliZ(0))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.circuit(*args, **kwds)


class Instructor:
    def __init__(self, n_qubits, n_layers) -> None:
        self.max_freq = n_qubits * n_layers
        self.model = Model(n_qubits, n_layers)

        rng = np.random.default_rng(100)

        x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
        omega_d = np.array([1, 1.2, 3])

        n_d = int(np.ceil(2 * np.max(np.abs(x_domain)) * np.max(omega_d)))
        print(f"Using {n_d} data points")
        self.x_d = np.linspace(x_domain[0], x_domain[1], n_d)

        self.weights = np.pi * (1 - 2 * rng.random(size=(n_layers, n_qubits * 3 - 1)))

    def forward(self, x_d, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        return self.model(self.weights, x_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

    def cost(self, w, x_d, y_d):
        y_pred = self.model(
            w, x_d, bf=self.bf, pf=self.pf, ad=self.ad, pd=self.pd, dp=self.dp
        )

        return np.mean((y_d - y_pred) ** 2)
