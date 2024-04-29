from typing import Any, Dict, Optional, Tuple, List

import pennylane as qml
import pennylane.numpy as np
from pennylane.fourier import coefficients
from functools import partial
from pennylane.fourier.visualize import _extract_data_and_labels
import hashlib
import os

from utils.ansaetze import Ansaetze


class Model:
    """
    A quantum circuit model.

    Parameters:
        n_qubits (int): The number of qubits in the circuit.
        n_layers (int): The number of layers in the circuit.
        circuit_type (str): The type of quantum circuit to use.
            If None, defaults to circuit19
        data_reupload (bool, optional): Whether to reupload data to the
            quantum device on each measurement. Defaults to True.
        tffm (bool, optional): Whether to use the TensorFlow Quantum
            Fourier Machine Learning interface. Defaults to False.
        state_vector (bool, optional): Whether to measure the state vector
            instead of the wave function. Defaults to False.

    Attributes:
        n_qubits (int): The number of qubits in the circuit.
        n_layers (int): The number of layers in the circuit.
        state_vector (bool): Whether to measure the state vector instead of
            the wave function.
        data_reupload (bool): Whether to reupload data to the quantum device
            on each measurement.
        tffm (bool): Whether to use the TensorFlow Quantum Fourier Machine
            Learning interface.
        pqc (Callable[[np.ndarray], None]): A callable function that applies
            the quantum circuit to the provided parameters.
        n_params (int): The number of parameters in the circuit.
        dev (qml.Device): The quantum device to use for the circuit.
        circuit (qml.QNode): The quantum circuit as a QNode.

    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        circuit_type: str,
        data_reupload: bool = True,
        tffm: bool = False,
        state_vector: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_vector = state_vector
        self.data_reupload = data_reupload
        self.tffm = tffm
        if circuit_type is None:
            circuit_type = "no_ansatz"
        self.pqc = getattr(Ansaetze, circuit_type)

        self.n_params = (self.n_layers, self.pqc(None, self.n_qubits))

        self.dev = qml.device("default.mixed", wires=n_qubits)

        self.circuit = qml.QNode(self._circuit, self.dev)

    def iec(
        self,
        x: np.ndarray,
        data_reupload: bool = True,
    ) -> None:
        """
        Creates an AngleEncoding using RY gates

        Args:
            x (np.ndarray): length of vector must be 1
            data_reupload (bool): Whether to reupload the data for the IEC
                or not, default is True.
        """
        if data_reupload:
            for q in range(self.n_qubits):
                qml.RX(x, wires=q)
        else:
            qml.RX(x, wires=0)

    def _circuit(
        self,
        w: np.ndarray,
        x: np.ndarray,
        bf: float = 0.0,
        pf: float = 0.0,
        ad: float = 0.0,
        pd: float = 0.0,
        dp: float = 0.0,
    ) -> float:
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
            float: Expectation value of PauliZ(0) of the circuit.
        """
        # assert isinstance(w, list) or w.shape == self.n_params, (
        #     "Number of parameters do not match. "
        #     f"Expected parameters of shape {self.n_params}, got {w.shape}"
        # )
        for l in range(0, self.n_layers - 1):
            self.pqc(w[l], self.n_qubits)

            if self.data_reupload or l == 0:
                self.iec(x, data_reupload=self.data_reupload)

            for q in range(self.n_qubits):
                qml.BitFlip(bf, wires=q)
                qml.PhaseFlip(pf, wires=q)
                qml.AmplitudeDamping(ad, wires=q)
                qml.PhaseDamping(pd, wires=q)
                qml.DepolarizingChannel(dp, wires=q)
        self.pqc(w[-1], self.n_qubits)

        if self.state_vector:
            return qml.state()
        else:
            return qml.expval(qml.PauliZ(wires=0))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.circuit(*args, **kwds)


class Instructor:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        seed: int = 100,
        circuit_type: int = 19,
        data_reupload: bool = True,
        tffm: bool = False,
        **kwargs,
    ) -> None:
        """
        Constructor for the Instructor class.

        Args:
            n_qubits: Number of qubits to use in the instructor circuit.
            n_layers: Number of layers in the instructor circuit.
            seed: Random seed to use for weight initialization.
            circuit_type: Type of circuit to use as the instructor.
            data_reupload: Whether or not to reupload data in the circuit.
            tffm: Whether or not to use trainable frequency feature mapping.
        """
        self.max_freq = n_qubits * n_layers

        if data_reupload:
            impl_n_layers = n_layers + 1  # we need L+1 according to Schuld et al.
        else:
            impl_n_layers = n_layers

        self.model = Model(
            n_qubits,
            impl_n_layers,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
            tffm=tffm,
            **kwargs,
        )

        rng = np.random.default_rng(seed)

        x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
        omega_d = np.array([1, 2, 3])

        n_d = int(np.ceil(2 * np.max(np.abs(x_domain)) * np.max(omega_d)))
        self.x_d = np.linspace(x_domain[0], x_domain[1], n_d)

        y_fct = lambda x: 1 / np.linalg.norm(omega_d) * np.sum(np.cos(omega_d * x))
        self.y_d = np.array([y_fct(x) for x in self.x_d])

        # self.weights = 2 * np.pi * rng.random(size=(impl_n_layers, n_qubits * 3))
        self.weights = 2 * np.pi * rng.random(size=self.model.n_params)

        self.opt = qml.AdamOptimizer(stepsize=0.01)

    def calc_hist(
        self,
        w: np.ndarray,
        bf: float = 0.0,
        pf: float = 0.0,
        ad: float = 0.0,
        pd: float = 0.0,
        dp: float = 0.0,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Calculate the histogram of the coefficients of the circuit output.

        Args:
            w: The weights of the quantum circuit.
            bf: The bit flip rate.
            pf: The phase flip rate.
            ad: The amplitude damping rate.
            pd: The phase damping rate.
            dp: The depolarization rate.

        Returns:
            A tuple containing the length of the histogram and a dictionary with the
            histogram data. The dictionary has the keys "real" and "imag" containing
            the real and imaginary part of the coefficients respectively, and "comb"
            containing the combination of the two.
        """
        coeffs = coefficients(
            partial(self.forward, weights=w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp),
            1,
            self.max_freq,
        )
        nvecs_formatted, data = _extract_data_and_labels(np.array([coeffs]))
        data_len = len(data["real"][0])
        """Calculate the histogram of the coefficients of the circuit output.

        Args:
            w (np.ndarray): The weights of the quantum circuit.
            bf (float): The bit flip rate.
            pf (float): The phase flip rate.
            ad (float): The amplitude damping rate.
            pd (float): The phase damping rate.
            dp (float): The depolarization rate.

        Returns:
            A tuple of (int, Dict[str, np.ndarray]): The length of the histogram and a
            dictionary with the histogram data. The dictionary has the keys "real" and
            "imag" containing the real and imaginary part of the coefficients
            respectively, and "comb" containing the combination of the two.
        """
        data["comb"] = np.sqrt(data["real"] ** 2 + data["imag"] ** 2)

        return data_len, data

    def forward(
        self,
        x_d: np.ndarray,
        weights: Optional[np.ndarray] = None,
        *,
        bf: float = 0.0,
        pf: float = 0.0,
        ad: float = 0.0,
        pd: float = 0.0,
        dp: float = 0.0,
        cache=False,
    ) -> np.ndarray:
        """Perform a forward pass of the quantum circuit.

        Args:
            x_d (np.ndarray): The input data.
            weights (Optional[np.ndarray], optional): The weights of the quantum
                circuit. If None, uses the current weights of the Instructor instance.
                Defaults to None.
            bf (float, optional): The bit flip rate. Defaults to 0.0.
            pf (float, optional): The phase flip rate. Defaults to 0.0.
            ad (float, optional): The amplitude damping rate. Defaults to 0.0.
            pd (float, optional): The phase damping rate. Defaults to 0.0.
            dp (float, optional): The depolarization rate. Defaults to 0.0.
            cache (bool, optional): Whether to cache the results. Defaults to False.

        Returns:
            np.ndarray: The output of the quantum circuit.

        """
        if weights is not None:
            w = weights
        else:
            w = self.weights

        if type(w) == list:
            w = np.array(w)

        # the qasm representation contains the bound parameters, thus it is ok to hash that
        hs = hashlib.md5(
            repr(
                {
                    "n_qubits": self.model.n_qubits,
                    "n_layers": self.model.n_layers,
                    "pqc": self.model.pqc.__name__,
                    "dru": self.model.data_reupload,
                    "w": w,
                    "x": x_d,
                    "bf": bf,
                    "pf": pf,
                    "ad": ad,
                    "pd": pd,
                    "dp": dp,
                }
            ).encode("utf-8")
        ).hexdigest()

        if cache:
            name = f"pqc_{hs}.npy"

            cache_folder = ".cache"
            if not os.path.exists(cache_folder):
                os.mkdir(cache_folder)

            file_path = os.path.join(cache_folder, name)

            if os.path.isfile(file_path):
                loaded_array = np.load(file_path)
                return loaded_array

        # execute the PQC circuit with the current set of parameters
        # ansatz = circuit(params, circuit.num_qubits)
        result = self.model(w, x_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

        if cache:
            np.save(file_path, result)

        return result

    def cost(
        self,
        w: np.ndarray,
        y_d: np.ndarray,
        bf: float = 0.0,
        pf: float = 0.0,
        ad: float = 0.0,
        pd: float = 0.0,
        dp: float = 0.0,
    ) -> float:
        """Compute the cost of the model.

        Args:
            w (np.ndarray): The weights of the quantum circuit.
            y_d (np.ndarray): The true labels of the input data.
            bf (float, optional): The bit flip rate. Defaults to 0.0.
            pf (float, optional): The phase flip rate. Defaults to 0.0.
            ad (float, optional): The amplitude damping rate. Defaults to 0.0.
            pd (float, optional): The phase damping rate. Defaults to 0.0.
            dp (float, optional): The depolarization rate. Defaults to 0.0.

        Returns:
            float: The cost of the model.
        """
        y_pred = self.forward(self.x_d, weights=w, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp)

        return np.mean((y_d - y_pred) ** 2)

    def step(
        self,
        w: List[float],
        bf: float = 0.0,
        pf: float = 0.0,
        ad: float = 0.0,
        pd: float = 0.0,
        dp: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """Perform a single optimization step.

        Args:
            w (List[float]): The weights of the quantum circuit.
            bf (float, optional): The bit flip rate. Defaults to 0.0.
            pf (float, optional): The phase flip rate. Defaults to 0.0.
            ad (float, optional): The amplitude damping rate. Defaults to 0.0.
            pd (float, optional): The phase damping rate. Defaults to 0.0.
            dp (float, optional): The depolarization rate. Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, float]: The updated weights and the cost of the model.
        """
        if len(w) == 0:
            w = self.weights

        w = np.array(w, requires_grad=True)

        w, cost = self.opt.step_and_cost(
            self.cost, w, y_d=self.y_d, bf=bf, pf=pf, ad=ad, pd=pd, dp=dp
        )

        return w, cost
