import pennylane as qml
from itertools import product
from time import time

import numpy as np

rng = np.random.default_rng(1111)

n_qubits = 10
n_layers = 10
max_freq = n_qubits * n_layers

dev = qml.device('default.qubit', wires=n_qubits)

weights = rng.random(size=(n_layers, n_qubits, 2))
x = 0.1

@qml.qnode(dev)
def circuit_with_weights(w, x):
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RX(x, wires=q)

            qml.RY(w[l, q, 0], wires=q)
            qml.RZ(w[l, q, 1], wires=q)

        if n_qubits > 1:
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

    return qml.expval(qml.PauliZ(0))


def get_coefficients(circuit_func, max_freq, weights):
    r"""Function taken from Pennylane and modified to only compute half of the Fourier coefficients.
    Only one half of the frequency spectrum suffices as the coefficient :math:`c_{\omega}`
    is equal to :math:`(c_{-\omega})*`, where :math:`\omega` cooresponds to one frequency and c to its
    coefficient.

    Computes the first :math:`d+1` Fourier coefficients of a :math:`2\pi` periodic
    function, where :math:`d` is the highest desired frequency in the Fourier spectrum.

    This function computes the coefficients blindly without any filtering applied, and
    is thus used as a helper function for the true ``coefficients`` function.

    Args:
        circuit_func (callable): function that takes weights and a 1D array of scalar inputs
        max_freq (int or tuple[int]): max frequency of Fourier coeffs to be computed. For degree
            :math:`d`, the coefficients from frequencies :math:`0,..., d-1, d`
            will be computed.
        weights (np.array or list): list of weig

    Returns:
        array[complex]: The Fourier coefficients of the function f up to the specified degree.
    """
    if isinstance(max_freq, int):
        max_freq = [max_freq]

    degree = np.array(max_freq)

    # number of integer values for the indices n_i = -degree_i,...,0,...,degree_i
    k = 2 * degree + 1

    # create generator for indices nvec = (n1, ..., nN), ranging from (-d1,...,-dN) to (d1,...,dN)
    n_ranges = [np.arange(d + 1) for d in degree]
    nvecs = product(*(n_ranges))

    # here we will collect the discretized values of function f
    f_discrete = np.zeros(shape=tuple(k))

    spacing = 2 * np.pi / k

    for nvec in nvecs:
        sampling_point = spacing * np.array(nvec)
        # fill discretized function array with value of f at inpts
        f_discrete[nvec] = circuit_func(w=weights, x=sampling_point)

    coeffs = np.fft.fftn(f_discrete) / k

    return coeffs

def fourier_series(coefficients, frequencies, x):
    fs_sum = 0
    for c, omega in zip(coefficients, frequencies):
        print(f"c_{omega} = (c_{-omega})* = {c}")
        fs_sum += c * np.exp(1j * x * omega)
        if omega > 0:
            fs_sum += np.conjugate(c) * np.exp(1j * x * -omega)
    return fs_sum

def main():

    start = time()

    # freqs = np.concatenate([np.arange(max_freq+1), np.arange(-max_freq, 0)])
    freqs = np.arange(max_freq + 1)
    coeffs = get_coefficients(circuit_with_weights, max_freq, weights)

    res_fft = fourier_series(coeffs, np.array(freqs), x)
    end = time()
    print(f"Calculation using FFT took {end-start}")
    print(f"Result using FFT: {np.real(res_fft)}")

    start = time()
    res_pl = circuit_with_weights(weights, x)
    end = time()
    print(f"Calculation using pennylane took {end-start}")
    print(f"Result using Pennylane: {res_pl}")

if __name__ == "__main__":
    main()
