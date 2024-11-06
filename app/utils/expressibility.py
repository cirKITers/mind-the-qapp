from .instructor import Instructor
import pennylane.numpy as np
from typing import Tuple, Dict, Optional
from scipy import integrate
import os


def theoretical_haar_probability(fidelity: float, n_qubits: int) -> float:
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


def sampled_haar_probability(n_qubits: int, n_bins: int) -> np.ndarray:
    """
    Calculates theoretical probability density function for random Haar states
    as proposed by Sim et al. (https://arxiv.org/abs/1905.10876) and bins it
    into a 2D-histogram.

    :param n_qubits: int: number of qubits in the quantum system
    :param n_bins: int: number of histogram bins
    :return: float: probability distribution for all fidelities
    """
    dist = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        l_idx = (1 / n_bins) * bin_idx
        u = l_idx + (1 / n_bins)
        dist[bin_idx], _ = integrate.quad(
            theoretical_haar_probability, l_idx, u, args=(n_qubits,)
        )

    return dist


def get_sampled_haar_probability_histogram(
    n_qubits, n_bins, n_repetitions, cache=True
) -> Tuple[np.ndarray, np.ndarray]:
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

    if cache:
        name = f"haar_{n_qubits}q_{n_bins}s.npy"

        cache_folder = ".cache"
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

        file_path = os.path.join(cache_folder, name)

        if os.path.isfile(file_path):
            y = np.load(file_path)
            return x, y

    # Note that this is a jax rng, so it does not matter if we
    # call that multiple times
    y = sampled_haar_probability(n_qubits, n_bins)

    if cache:
        np.save(file_path, y)

    return x, y


def get_kl_divergence_expr(
    vqc_prob_dist: np.ndarray,
    haar_dist: np.ndarray,
) -> np.ndarray:
    """
    Calculates the KL divergence between two probability distributions (Haar
    probability distribution and the fidelity distribution sampled from a VQC).

    :param vqc_prob_dist: np.ndarray: VQC fidelity probability distribution.
        Should have shape (n_inputs_samples, n_bins)
    :param haar_dist: np.ndarray: Haar probability distribution with shape.
        Should have shape (n_bins, )
    :return: np.ndarray: Array of KL-Divergence values for all values in axis 1
    """
    assert all([haar_dist.shape == p.shape for p in vqc_prob_dist]), (
        "All "
        "probabilities for inputs should have the same shape as Haar. "
        f"Got {haar_dist.shape} for Haar and {vqc_prob_dist.shape} for VQC"
    )

    kl_divergence = np.zeros(vqc_prob_dist.shape[0])
    for i, p in enumerate(vqc_prob_dist):
        kl_divergence[i] = np.sum(np.where(p != 0, p * np.log(p / haar_dist), 0))

    return kl_divergence


class Expressibility_Sampler:
    def __init__(
        self,
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

        self.instructor = Instructor(
            n_qubits,
            n_layers,
            seed=seed,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
        )
        self.rng = np.random.default_rng(seed)

        self.epsilon = 1e-5

        x_domain = [-1 * np.pi, 1 * np.pi]
        self.x_samples = np.linspace(
            x_domain[0], x_domain[1], n_input_samples, requires_grad=False
        )

    def sample_state_fidelities(
        self,
        noise_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:

        fidelities = np.zeros((len(self.x_samples), self.n_samples))

        # moving the rng out of for loops can improve performance
        # python can handle large arrays pretty well
        # but calls to the rng are relatively slow
        w = (
            2
            * np.pi
            * (
                1
                - 2
                * self.rng.random(
                    size=[*self.instructor.model.params.shape, self.n_samples * 2]
                )
            )
        )

        x_samples_batched = self.x_samples.reshape(1, -1).repeat(
            self.n_samples * 2, axis=0
        )

        for idx in range(len(self.x_samples)):

            sv = self.instructor.model(
                params=w,
                inputs=x_samples_batched[:, idx],
                noise_params=noise_params,
                cache=True,
                execution_type="density",
            )  # n_samples, N
            sqrt_sv1 = np.sqrt(sv[: self.n_samples])

            fidelity = (
                np.trace(
                    np.sqrt(sqrt_sv1 * sv[self.n_samples :] * sqrt_sv1),
                    axis1=1,
                    axis2=2,
                )
                ** 2
            )
            fidelities[idx] = fidelity
        fidelities[-1] = fidelities[0]  # fist input is the equal to the last one

        return fidelities

    def sample_hist_state_fidelities(
        self, noise_params: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fidelities = self.sample_state_fidelities(noise_params)
        z_component = np.zeros((len(self.x_samples), self.n_bins))

        # FIXME: somehow I get nan's in the histogram,
        # when directly creating bins until n
        # workaround hack is to add a small epsilon
        # could it be related to sampling issues?
        b = np.linspace(0, 1 + self.epsilon, self.n_bins + 1)
        for i, f in enumerate(fidelities):
            z_component[i], _ = np.histogram(f, bins=b)
        z_component = np.transpose(z_component) / self.n_samples
        return self.x_samples, b, z_component
