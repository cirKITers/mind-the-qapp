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
    ) -> None:
        """
        Constructor for the EntanglingCapability_Sampler class.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the quantum circuit.
        n_layers : int
            Number of layers in the quantum circuit.
        seed : int, optional
            Seed for the NumPy RNG, by default 100
        circuit_type : int, optional
            Type of quantum circuit to use, by default 19
        data_reupload : bool, optional
            Whether to reupload the data to the QPU, by default True
        """
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

    def calculate_entangling_capability(self, samples_per_qubit: int) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Parameters
        ----------
        samples_per_qubit : int
            Number of samples per qubit.

        Returns
        -------
        float
            Entangling capacity of the given circuit.
        """

        def meyer_wallach(n_qubits: int, samples: int, params_shape: tuple, rng):
            """
            Calculates the Meyer-Wallach sampling of the entangling capacity
            of a quantum circuit.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.
            samples : int
                Number of samples to be taken.
            params_shape : tuple
                Shape of the parameter vector.
            rng : np.random.default_rng
                Random number generator.

            Returns
            -------
            float
                Entangling capacity of the given circuit.
            """
            mw_measure = np.zeros(samples, dtype=complex)

            params = rng.uniform(0, 2 * np.pi, size=(samples, *params_shape))

            qb = list(range(n_qubits))

            for i in range(samples):
                U = self.instructor.forward(0, params[i])

                qb = list(range(n_qubits))
                entropy = 0

                for j in range(n_qubits):
                    density = qml.math.partial_trace(U, qb[:j] + qb[j + 1 :])
                    entropy += np.trace(density**2)

                entropy = min((entropy.real / n_qubits), 1)
                mw_measure[i] = 1 - entropy

            return 2 * np.sum(mw_measure).real / samples

        circuit = self.instructor.model.circuit
        # TODO: propagate precision to kedro parameters
        entangling_capability = meyer_wallach(
            n_qubits=len(circuit.device.wires),
            samples=samples_per_qubit * len(circuit.device.wires),
            params_shape=self.instructor.model.n_params,
            rng=self.rng,
        )
        return float(entangling_capability)
