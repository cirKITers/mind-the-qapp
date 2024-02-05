from typing import Any

import pennylane as qml
import pennylane.numpy as np
from pennylane.fourier import coefficients
from functools import partial
from pennylane.fourier.visualize import _extract_data_and_labels


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

        self.steps = 10
        rng = np.random.default_rng(200)

        x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
        omega_d = np.array([1, 2, 3, 4])

        n_d = int(np.ceil(2 * np.max(np.abs(x_domain)) * np.max(omega_d)))
        self.x_d = np.linspace(x_domain[0], x_domain[1], n_d)

        y_fct = lambda x: 1 / np.linalg.norm(omega_d) * np.sum(np.cos(omega_d * x))
        self.y_d = np.array([y_fct(x) for x in self.x_d])

        self.weights = (
            2 * np.pi * (1 - 2 * rng.random(size=(n_layers, n_qubits * 3 - 1)))
        )

        self.opt = qml.GradientDescentOptimizer(stepsize=0.05)

        self.clear_hist()

    def clear_hist(self):
        self.pred = []
        self.x = []
        self.y = []
        self.z = []
        self.loss = []

    def append_hist(self, hist):
        self.z.append(hist)
        if len(self.y) == 0:
            self.y.append(0)
        else:
            self.y.append(self.y[-1] + 1)
        return {"y": np.array(self.y), "z": np.array(self.z)}

    def get_hist(self):
        return {
            "x": np.array(self.x),
            "y": np.array(self.y),
            "z": np.array(self.z),
        }

    def calc_hist(self, w, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        coeffs = coefficients(
            partial(self.forward, weights=w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
            1,
            self.max_freq,
        )
        nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
        data_len = len(data["real"][0])
        data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

        self.x = np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1)

        return data_len, data

    def forward(self, x_d, weights=None, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        if weights is not None:
            w = weights
        else:
            w = self.weights
        return self.model(w, x_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

    def cost(self, w, y_d, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        y_pred = self.model(w, self.x_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

        return np.mean((y_d - y_pred) ** 2)

    def step(self, w, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        w, cost = self.opt.step_and_cost(
            self.cost, w, y_d=self.y_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )
        # grad, pred = opt.compute_grad(
        #     self.cost,
        #     w,
        #     grad_fn=None,
        #     kwargs=dict(
        #         y_d=self.y_d,
        #         bf=bf,
        #         pf=pf,
        #         ad=ad,
        #         pd=pd,
        #         dp=dp,
        #     ),
        # )

        # w = opt.apply_grad(grad, w)

        return self.forward(self.x_d, w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp), w, cost
