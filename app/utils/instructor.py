from typing import Dict, Optional, Tuple, List

import pennylane as qml
import pennylane.numpy as np

from qml_essentials.model import Model
from qml_essentials.coefficients import Coefficients
from qml_essentials.entanglement import Entanglement
from qml_essentials.expressibility import Expressibility

import logging

log = logging.getLogger(__name__)


class Instructor:
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        n_freqs: int = 3,
        stepsize: float = 0.01,
        seed: int = 100,
        circuit_type: str = "Circuit_19",
        data_reupload: bool = True,
        coefficients: Optional[List[float]] = None,
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
            coefficients: List of coefficients for the Fourier series. If None, uses 0.5 for all.
        """
        self.seed = seed
        self.stepsize = stepsize
        self.n_freqs = n_freqs
        self.circuit_type = circuit_type
        self.coefficients = (
            coefficients if coefficients is not None else [0.5] * n_freqs
        )

        self.model = Model(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
            random_seed=seed,
            **kwargs,
        )
        self.x_domain = [-1 * np.pi, 1 * np.pi]  # [-4 * np.pi, 4 * np.pi]

        self.x_d = self.sample_domain(self.x_domain)
        self.y_d = self.generate_fourier_series(self.x_d, n_freqs, self.coefficients)
        self.opt = qml.AdamOptimizer(stepsize=stepsize)

    def sample_domain(
        self,
        domain: List[float],
    ) -> np.ndarray:
        """
        Generates a evenly spaced grid of data points in a domain [min, max].

        Parameters
        ----------
        domain : List[float]
            Domain from which to sample.

        Returns
        -------
        np.ndarray
            Grid tensor with shape (> 2 * n_freqs,)
        """
        # We always have at least five data points
        n_d = max(int(np.ceil(2 * np.max(np.abs(domain))) * (self.n_freqs)), 5)

        log.info(f"Using {n_d} data points on {self.n_freqs} frequencies")

        grid = np.linspace(domain[0], domain[1], num=n_d)

        return grid

    def generate_fourier_series(
        self,
        domain_samples: np.ndarray,
        omegas: List[List[float]],
        coefficients: List[List[float]],
    ) -> np.ndarray:
        """
        Generates the Fourier series representation of a function.

        Parameters
        ----------
        domain_samples : np.ndarray
            Grid of domain samples.
        omega : List[List[float]]
            List of frequencies for each dimension.

        Returns
        -------
        np.ndarray
            Fourier series representation of the function.
        """
        if not isinstance(omegas, list):
            # plus one as zero freq does not count
            omegas = [o for o in range(omegas + 1)]
        if not isinstance(coefficients, list):
            coefficients = [coefficients for _ in omegas]

        assert len(omegas) == len(
            coefficients
        ), "Number of frequencies and coefficients must match"

        omegas = np.array(omegas)
        coefficients = np.array(coefficients)

        def y(x: np.ndarray) -> float:
            """
            Calculates the Fourier series representation of a function at a given point.

            Parameters
            ----------
            x : np.ndarray
                Point at which to evaluate the function.

            Returns
            -------
            float
                Value of the Fourier series representation at the given point.
            """
            return (
                1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))
            )  # transpose!

        values = np.stack([y(x) for x in domain_samples])

        return values

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
            scale=False,
        )

    def haar_integral(self, n_bins):
        return Expressibility().haar_integral(
            n_qubits=self.model.n_qubits, n_bins=n_bins, scale=False
        )

    def kullback_leibler(self, a, b):
        return Expressibility().kullback_leibler_divergence(a, b)

    def cost(
        self,
        params: np.ndarray,
        x_d: np.ndarray,
        y_d: np.ndarray,
        **kwargs,
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
            inputs=x_d,
            **kwargs,
        )

        return np.mean((y_d - y_pred) ** 2)

    def step(
        self,
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
        y_pred = self.model(
            params=self.model.params,
            inputs=self.x_d,
            noise_params=noise_params,
            cache=False,
            execution_type="expval",
            force_mean=True,
        )

        self.model.params, cost = self.opt.step_and_cost(
            self.cost,
            self.model.params,
            x_d=self.x_d,
            y_d=self.y_d,
            noise_params=noise_params,
            cache=False,
            execution_type="expval",
            force_mean=True,
        )

        return (cost, y_pred)
