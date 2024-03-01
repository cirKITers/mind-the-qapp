from .instructor import Model
import numpy as np
from typing import Tuple
from scipy import integrate

def theoretical_haar_probability(fidelity: float, n_qubits: int) \
        -> float:
    """
    Calculates theoretical probability density function for random Haar states
    as proposed by Sim et al. (https://arxiv.org/abs/1905.10876).

    :param fidelity: float: fidelity of two parameter assignments in [0, 1]
    :param n_qubits: int: number of qubits in the quantum system
    :return: float: probability for a given fidelity
    """
    N = 2**n_qubits

    prob = (N - 1) * (1 - fidelity) ** (N - 2)
    return prob

def sampled_haar_probability(n_qubits: int, n_bins: int) \
        -> np.ndarray:
    """
    Calculates theoretical probability density function for random Haar states
    as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
    into a 2D-histogram.

    :param n_qubits: int: number of qubits in the quantum system
    :param n_bins: int: number of histogram bins
    :return: float: probability distribution for all fidelities
    """
    dist = np.zeros(n_bins)
    for i in range(n_bins):
        l = (1/n_bins) * i
        u = l + (1/n_bins)
        dist[i], _ = integrate.quad(theoretical_haar_probability, l, u, args=(n_qubits,))

    return dist

def get_sampled_haar_probability_histogram(n_qubits, n_bins, n_repetitions) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculates theoretical probability density function for random Haar states
    as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
    into a 3D-histogram.

    :param n_qubits: int: number of qubits in the quantum system
    :param n_bins: int: number of histogram bins
    :param n_repetitions: int: number of repetitions for the x-axis
    :return: np.ndarray: x component (bins)
    :return: np.ndarray: y component (probabilities)
    """
    x = np.linspace(0, 1, n_bins)
    y = sampled_haar_probability(n_qubits, n_bins)

    return x, y

class Expressibility_Sampler:
    def __init__(self,
        n_qubits: int,
        n_layers: int,
        seed: int = 100,
        circuit_type: int = 19,
        data_reupload: bool = True,
        n_samples: int = 1000,
        n_input_samples: int = 10,
        n_bins: int = 75,
    ) -> None:

        self.n_samples = n_samples
        self.n_bins = n_bins

        self.model = Model(
            n_qubits,
            n_layers,
            circuit_type,
            data_reupload=data_reupload,
            state_vector=True,
        )
        self.rng = np.random.default_rng(seed)

        self.epsilon = 1e-5

        x_domain = [-1 * np.pi, 1 * np.pi]
        self.x_samples = np.linspace(x_domain[0], x_domain[1], n_input_samples)

    def sample_state_fidelities(self) -> np.ndarray:

        fidelities = np.zeros((len(self.x_samples), self.n_samples))
        for i, x in enumerate(self.x_samples):
            for s in range(self.n_samples):

                w1 = 2 * np.pi * (1 - 2 * self.rng.random(size=self.model.n_params))
                sv1 = self.model(w1, x)

                w2 = 2 * np.pi * (1 - 2 * self.rng.random(size=self.model.n_params))
                sv2 = self.model(w2, x)

                fidelity = np.trace(np.sqrt(np.sqrt(sv1) * sv2 * np.sqrt(sv1)))**2
                fidelity = np.abs(fidelity)

                fidelities[i, s] = fidelity

        return fidelities

    def sample_hist_state_fidelities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fidelities = self.sample_state_fidelities()
        z_component = np.zeros((len(self.x_samples), self.n_bins-1))

        # FIXME: somehow I get nan's in the histogram, when directly creating bins until n
        # workaround hack is to add a small epsilon
        b = np.linspace(0, 1 + self.epsilon, self.n_bins)
        for i, x in enumerate(self.x_samples):
            z_component[i], _ = np.histogram(fidelities[i], bins=b, density=True)
        z_component = np.transpose(z_component)
        return self.x_samples, b, z_component

