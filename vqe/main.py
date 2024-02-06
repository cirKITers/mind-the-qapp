import pennylane as qml
import tensorflow as tf
from keras.optimizers import SGD
import numpy as np
from functools import partial
from pennylane.fourier import coefficients
from pennylane.fourier.visualize import bar
import matplotlib.pyplot as plt

# ignore tensorflow's complex->float cast warning
tf.get_logger().setLevel('ERROR')

import qubo_utils as qu
import circuit_utils as cu
import config as cfg
import random_utils as ru


def optimise_VQE():
    """
    Run VQE 
    """
    m_ising, offset = qu.provide_random_maxcut_ising(cfg.n_qubits, cfg.edge_prob, cfg.problem_seed)

    parameters = tf.Variable(tf.random.uniform((cfg.n_layers * 2 * cfg.n_qubits,)),
                             trainable=True)

    dev = qml.device("default.qubit")
    circ = qml.QNode(cu.circ_2, dev, interface="tf")

    observable = cu.ising_hamiltonian(m_ising)

    optimizer = SGD(cfg.learning_rate)

    costs = []

    for i in range(cfg.max_iterations):
        with tf.GradientTape() as tape:

            cost = circ(parameters, observable, cfg.n_layers) + offset

        grads = tape.gradient(cost, parameters)

        #print(f"Current cost at step {i+1}: {cost.numpy():.4f}")
        costs.append(cost)
        optimizer.apply_gradients([(grads, parameters)])

        if i > 0 and costs[-2] - costs[-1] < cfg.tol:
            break

    print(f"Reached a minimum cost of {costs[-1].numpy():.4f} after {i+1} optimisation steps.")
    return costs, parameters

if __name__ == "__main__":
    ru.set_global_seed(cfg.global_seed)
    costs, parameters = optimise_VQE()

    m_ising, offset = qu.provide_random_maxcut_ising(cfg.n_qubits, cfg.edge_prob, cfg.problem_seed)
    observable = cu.ising_hamiltonian(m_ising)

    dev = qml.device("default.qubit")
    circ = qml.QNode(cu.circ_2, dev, interface="tf")

    f = partial(circ, observable = observable, n_layers = cfg.n_layers)

    print(f(parameters) + offset)

    coeffs = coefficients(f, cfg.n_layers*cfg.n_qubits*2, 2)
    print(coeffs)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(50, 4))
    bar(coeffs, cfg.n_layers * cfg.n_qubits * 2, ax, colour_dict={"real" : "red", "imag" : "blue"})
    plt.savefig('coeffs.png')


