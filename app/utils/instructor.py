from typing import Dict, Optional, Tuple, List

import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients
from qml_essentials.entanglement import Entanglement
from qml_essentials.expressibility import Expressibility


class Instructor:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_freqs: int,
        seed: int = 100,
        circuit_type: int = 19,
        data_reupload: bool = True,
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
        self.seed = seed

        self.model = Model(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
            random_seed=seed,
            **kwargs,
        )
        self.x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]
        omega_d = np.array([i for i in range(1, n_freqs + 1)])

        n_d = int(np.ceil(2 * np.max(np.abs(self.x_domain)) * np.max(omega_d)))
        self.x_d = np.linspace(
            self.x_domain[0], self.x_domain[1], n_d, requires_grad=False
        )

        def y_fct(x):
            return 1 / np.linalg.norm(omega_d) * np.sum(np.cos(omega_d * x))

        self.y_d = np.array([y_fct(x) for x in self.x_d], requires_grad=False)

        self.opt = qml.AdamOptimizer(stepsize=0.01)

    def calc_hist(
        self,
        params: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Calculate the histogram of the coefficients of the circuit output.

        Args:
            p: The params of the quantum circuit.
            bf: The bit flip rate.
            pf: The phase flip rate.
            ad: The amplitude damping rate.
            pd: The phase damping rate.
            dp: The depolarization rate.

        Returns:
            A tuple containing the length of the histogram and a dictionary with the
            histogram data.
        """
        self.model.params = params
        data = (
            Coefficients()
            .sample_coefficients(self.model, noise_params=noise_params, cache=False)
            .real
        )

        # rearange data such that the zero coefficient from the first
        # index goes to the center of the array
        data = np.array(
            [*data[1 : len(data) // 2 + 1], data[0], *data[len(data) // 2 + 1 :]]
        )
        return np.abs(data)

    def meyer_wallach(
        self,
        n_samples: Optional[int | None] = None,
        noise_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Calculate the entanglement of the circuit output.

        Args:
            p: The params of the quantum circuit.
            bf: The bit flip rate.
            pf: The phase flip rate.
            ad: The amplitude damping rate.
            pd: The phase damping rate.
            dp: The depolarization rate.

        Returns:
            A tuple containing the length of the histogram and a dictionary with the
            histogram data.
        """
        return Entanglement().meyer_wallach(
            self.model, n_samples=n_samples, seed=self.seed, noise_params=noise_params
        )

    def state_fidelities(
        self,
        n_samples,
        n_bins,
        n_input_samples,
        noise_params: Optional[Dict[str, float]] = None,
    ):
        return Expressibility().state_fidelities(
            seed=self.seed,
            n_samples=n_samples,
            n_bins=n_bins,
            n_input_samples=n_input_samples,
            input_domain=self.x_domain,
            model=self.model,
            noise_params=noise_params,
        )

    def haar_integral(self, n_bins):
        return Expressibility().haar_integral(
            n_qubits=self.model.n_qubits, n_bins=n_bins
        )

    def kullback_leibler(self, a, b):
        return Expressibility().kullback_leibler_divergence(a, b)

    def cost(
        self,
        params: np.ndarray,
        y_d: np.ndarray,
        noise_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute the cost of the model.

        Args:
            p (np.ndarray): The params of the quantum circuit.
            y_d (np.ndarray): The true labels of the input data.
            bf (float, optional): The bit flip rate. Defaults to 0.0.
            pf (float, optional): The phase flip rate. Defaults to 0.0.
            ad (float, optional): The amplitude damping rate. Defaults to 0.0.
            pd (float, optional): The phase damping rate. Defaults to 0.0.
            dp (float, optional): The depolarization rate. Defaults to 0.0.

        Returns:
            float: The cost of the model.
        """
        y_pred = self.model(
            params=params,
            inputs=self.x_d,
            noise_params=noise_params,
            cache=False,
            execution_type="expval",
        )

        return np.mean((y_d - y_pred) ** 2)

    def step(
        self,
        params: List[float],
        noise_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Perform a single optimization step.

        Args:
            p (List[float]): The params of the quantum circuit.
            bf (float, optional): The bit flip rate. Defaults to 0.0.
            pf (float, optional): The phase flip rate. Defaults to 0.0.
            ad (float, optional): The amplitude damping rate. Defaults to 0.0.
            pd (float, optional): The phase damping rate. Defaults to 0.0.
            dp (float, optional): The depolarization rate. Defaults to 0.0.

        Returns:
            Tuple[np.ndarray, float]: The updated params and the cost of the model.
        """
        if len(params) == 0:
            params = self.model.params

        y_pred = self.model(
            params=params,
            inputs=self.x_d,
            noise_params=noise_params,
            cache=False,
            execution_type="expval",
        )
        params = np.array(params, requires_grad=True)

        params, cost = self.opt.step_and_cost(
            self.cost, params, y_d=self.y_d, noise_params=noise_params
        )
        return (params, cost, y_pred[0])
