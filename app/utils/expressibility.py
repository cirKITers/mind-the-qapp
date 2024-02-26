from .instructor import Model
import numpy as np
from typing import Tuple

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

    return (N - 1) * (1 - fidelity) ** (N - 2)

def sampled_haar_probability(fidelity: float, n_qubits: int) \
        -> float:
    """
    Calculates theoretical probability density function for random Haar states
    as proposed by Sim et al. (https://arxiv.org/abs/1905.10876).

    :param fidelity: float: fidelity of two parameter assignments in [0, 1]
    :param n_qubits: int: number of qubits in the quantum system
    :return: float: probability for a given fidelity
    """
    N = 2**n_qubits

    return (N - 1) * (1 - fidelity) ** (N - 2)

class Expressibility_Sampler:
    def __init__(self,
                 n_qubits: int,
                 n_layers: int,
                 seed: int = 100,
                 circuit_type: int = 19,
                 n_samples: int = 1000,
                 n_input_samples: int = 10,
                 n_bins: int = 75,
                 ):

        self.n_samples = n_samples
        self.n_bins = n_bins

        self.model = Model(n_qubits, n_layers, circuit_type, state_vector=True)
        self.rng = np.random.default_rng(seed)

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

                fidelities[i, s] = np.sum(np.abs(np.conj(sv1.T) * sv2))
                print(s)

        return fidelities

    def sample_hist_state_fidelities(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fidelities = self.sample_state_fidelities()
        z_component = np.zeros((len(self.x_samples), self.n_bins-1))
        f = np.linspace(0, 1, self.n_bins)
        for i, x in enumerate(self.x_samples):
            z_component[i], _ = np.histogram(fidelities[i], bins=f, density=True)
        return self.x_samples, f, z_component

