import pennylane as qml
from pennylane import numpy as np
from typing import Dict
from .instructor import Instructor


class EntanglingCapability_Sampler:

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

        self.seed = seed
        self.instructor = Instructor(
            n_qubits,
            n_layers,
            seed=seed,
            circuit_type=circuit_type,
            data_reupload=data_reupload,
            state_vector=True,
        )
        self.rng = np.random.default_rng(seed)

        self.epsilon = 1e-5

        x_domain = [-1 * np.pi, 1 * np.pi]
        self.x_samples = np.linspace(
            x_domain[0], x_domain[1], n_input_samples, requires_grad=False
        )

    def calculate_entangling_capability(
        self, samples_per_qubit: int
    ) -> Dict[str, float]:
        """
        Calculate the entangling capability of a quantum circuit.
        The strategy is taken from https://doi.org/10.48550/arXiv.1905.10876
        Implementation inspiration from
        https://obliviateandsurrender.github.io/blogs/expr.html

        Sanity Check; The following circuit should yield an entangling capability of 1.
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0,1)
        qc.rx(*ParameterVector("x", 1), 0)

        Note that if qubits are being measured, this will return almost zero
        because the statevector simulator cannot work on collapsed states.

        If the circuit doesn't contain parameterizable operations, nan is returned.

        Args:
            circuit (QuantumCircuit): The quantum circuit.
            samples_per_qubit (int): The number of samples to generate.
            seed (int): The seed for the random number generator.

        Returns:
            dict: A dictionary containing the entangling capability value.
        """

        def meyer_wallach(n_qubits: int, samples, params_shape, rng):
            mw_measure = np.zeros(samples, dtype=complex)

            # FIXME: unify the range for parameters in the circuit
            # generation method and the sampling here
            params = rng.uniform(0, 2 * np.pi, size=(samples, *params_shape))

            # generate a list from [0..op_wires-1]
            # we need that later to trace out the corresponding qubits
            qb = list(range(n_qubits))

            # outer sum of the MW measure; iterate over set of parameters
            for i in range(samples):
                # execute the PQC circuit with the current set of parameters
                U = self.instructor.forward(0, params[i])

                # generate a list from [0..op_wires-1]
                # we need that later to trace out the corresponding qubits
                qb = list(range(n_qubits))
                # initialize the inner sum which corresponds to the entropy
                entropy = 0

                # inner sum of the MW measure
                for j in range(n_qubits):
                    # density of the jth qubit after tracing out the rest
                    density = qml.math.partial_trace(U, qb[:j] + qb[j + 1 :])
                    # trace of the density matrix
                    entropy += np.trace(density**2)

                # fixes accumulating decimals that would otherwise lead to a MW > 1
                entropy = min((entropy.real / n_qubits), 1)
                # inverse of the normalized entropy is the MW
                # for the current sample of parameters
                mw_measure[i] = 1 - entropy
            # final normalization according to formula
            return 2 * np.sum(mw_measure).real / samples

        rng = np.random.default_rng(seed=self.seed)

        circuit = self.instructor.model.circuit
        # TODO: propagate precision to kedro parameters
        entangling_capability = meyer_wallach(
            n_qubits=len(circuit.device.wires),
            samples=samples_per_qubit * len(circuit.device.wires),
            params_shape=self.instructor.model.n_params,
            rng=rng,
        )

        return float(entangling_capability)
