from typing import Any

import pennylane as qml
import pennylane.numpy as np
from pennylane.fourier import coefficients
from functools import partial
from pennylane.fourier.visualize import _extract_data_and_labels


class Model:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: int = 19,
        data_reupload=True,
        tffm=False,
        state_vector=False,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_vector = state_vector
        self.data_reupload = data_reupload
        self.tffm = tffm
        self.pqc = getattr(self, f"_pqc{circuit_type}")
        self.n_params = getattr(self, f"_n_params_circ{circuit_type}")()

        self.dev = qml.device("default.mixed", wires=n_qubits)

        self.circuit = qml.QNode(self._circuit, self.dev)

    # FIXME outsource num params and circuit definitions
    def _n_params_circ19(self) -> tuple:
        """
        Returns the number of Parameters for the Circuit19 Ansatz
        """
        return (self.n_layers, self.n_qubits * 3 - 1)

    def _pqc19(self, w: np.ndarray):
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

        for q in range(self.n_qubits):
            if q > 0:
                qml.CRX(w[w_idx], wires=[(q + 1) % self.n_qubits, q])
                w_idx += 1

    def _strongly_entangling(self, w: np.ndarray):
        qml.StronglyEntanglingLayers(w, wires=range(self.n_qubits))

    def iec(self, x: np.ndarray, data_reupload=True):
        """
        Creates an AngleEncoding using RY gates

        Args:
            x (np.ndarray): length of vector must be 1
        """
        if data_reupload:
            for q in range(self.n_qubits):
                qml.RX(x, wires=q)
        else:
            qml.RX(x, wires=0)

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
        # assert isinstance(w, list) or w.shape == self.n_params, (
        #     "Number of parameters do not match. "
        #     f"Expected parameters of shape {self.n_params}, got {w.shape}"
        # )
        for l in range(0, self.n_layers - 1):
            self.pqc(w[l].reshape(1, self.n_qubits, 3))
            if self.data_reupload or l == 0:
                self.iec(x, data_reupload=self.data_reupload)

            for q in range(self.n_qubits):
                qml.BitFlip(bf, wires=q)
                qml.PhaseFlip(pf, wires=q)
                qml.AmplitudeDamping(ad, wires=q)
                qml.PhaseDamping(pd, wires=q)
                qml.DepolarizingChannel(dp, wires=q)
        self.pqc(w[-1].reshape(1, self.n_qubits, 3))

        if self.state_vector:
            return qml.state()
        else:
            return qml.expval(qml.PauliZ(wires=0))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.circuit(*args, **kwds)


class Instructor:
    def __init__(
        self,
        n_qubits,
        n_layers,
        seed=100,
        circuit_type=19,
        data_reupload=True,
        tffm=False,
    ) -> None:
        self.max_freq = n_qubits * n_layers

        impl_n_layers = n_layers + 1  # we need L+1 according to Schuld et al.
        self.model = Model(
            n_qubits,
            impl_n_layers,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
            tffm=tffm,
        )

        rng = np.random.default_rng(seed)

        x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
        omega_d = np.array([1, 2, 3])

        n_d = int(np.ceil(2 * np.max(np.abs(x_domain)) * np.max(omega_d)))
        self.x_d = np.linspace(x_domain[0], x_domain[1], n_d)

        y_fct = lambda x: 1 / np.linalg.norm(omega_d) * np.sum(np.cos(omega_d * x))
        self.y_d = np.array([y_fct(x) for x in self.x_d])

        self.weights = 2 * np.pi * rng.random(size=(impl_n_layers, n_qubits * 3))

        self.opt = qml.AdamOptimizer(stepsize=0.01)

    def calc_hist(self, w, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        coeffs = coefficients(
            partial(self.forward, weights=w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
            1,
            self.max_freq,
        )
        nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
        data_len = len(data["real"][0])
        data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

        # self.x = np.arange(-data_len // 2 + 1, data_len // 2 + 1, 1)

        return data_len, data

    def forward(self, x_d, weights=None, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        if weights is not None:
            w = weights
        else:
            w = self.weights
        return self.model(w, x_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

    def cost(self, w, y_d, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        y_pred = self.forward(self.x_d, weights=w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

        return np.mean((y_d - y_pred) ** 2)

    def step(self, w, bf=0.0, pf=0.0, ad=0.0, pd=0.0, dp=0.0):
        if len(w) == 0:
            w = self.weights

        w = np.array(w, requires_grad=True)

        w, cost = self.opt.step_and_cost(
            self.cost, w, y_d=self.y_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )

        return w, cost
