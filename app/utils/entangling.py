import pennylane as qml
from pennylane import numpy as np
from typing import Dict, Optional
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
        )
        self.rng = np.random.default_rng(seed)

    def calculate_entangling_capability(
        self,
        samples_per_qubit: int,
        params: Optional[np.ndarray] = None,
        noise_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculates the entangling capacity of a given quantum circuit
        using Meyer-Wallach measure.

        Parameters
        ----------
        samples_per_qubit : int
            Number of samples per qubit.
        bf: float: The bit flip rate.
        pf: float: The phase flip rate.
        ad: float: The amplitude damping rate.
        pd: float: The phase damping rate.
        dp: float: The depolarization rate.
        params: optional, np.ndarray:
            Parameters of the instructor

        Returns
        -------
        float
            Entangling capacity of the given circuit.
        """

        def meyer_wallach(n_qubits: int, samples: int, params: np.ndarray):
            """
            Calculates the Meyer-Wallach sampling of the entangling capacity
            of a quantum circuit.

            Parameters
            ----------
            n_qubits : int
                Number of qubits in the circuit.
            samples : int
                Number of samples to be taken.
            params: optional, np.ndarray:
                Parameters of the instructor

            Returns
            -------
            float
                Entangling capacity of the given circuit.
            """
            mw_measure = np.zeros(samples, dtype=complex)

            qb = list(range(n_qubits))

            for i in range(samples):
                U = self.instructor.model(
                    inputs=None,
                    params=params[i],
                    noise_params=noise_params,
                    cache=True,
                    execution_type="density",
                )

                entropy = 0

                for j in range(n_qubits):
                    density = qml.math.partial_trace(U, qb[:j] + qb[j + 1 :])
                    entropy += np.trace((density @ density).real)

                mw_measure[i] = 1 - entropy / n_qubits

            mw = 2 * np.sum(mw_measure.real) / samples

            # catch floating point errors
            if mw < 0.0:
                mw = 0.0
            return mw

        circuit = self.instructor.model.circuit
        samples = samples_per_qubit * len(circuit.device.wires)

        if params is not None:
            assert params.shape == self.instructor.model.params.shape, (
                "Parameter shape of instructor, and that provided for "
                "entangling capability should be equal, but are "
                f"{params.shape} and {self.instructor.model.params.shape} "
                "respectively"
            )
            p = np.repeat(np.expand_dims(params, axis=0), samples, axis=0)
        else:
            p = self.rng.uniform(
                0, 2 * np.pi, size=(samples, *self.instructor.model.params.shape)
            )

        # TODO: propagate precision to kedro parameters
        entangling_capability = meyer_wallach(
            n_qubits=len(circuit.device.wires),
            samples=samples,
            params=p,
        )

        return float(entangling_capability)
