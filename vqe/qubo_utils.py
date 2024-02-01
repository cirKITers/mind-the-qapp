import networkx as nx
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf
import os
import pandas as pd
from itertools import product

# Global variable to store generated problems
# Prevents re-generation, when a problem was already generated
problems = dict()

def pure_ising_to_QUBO(J: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate Qubo Matrix Q and offset E0 from J, such that
    s^T J s equals x^T Q x + E0
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    :param J: np.ndarray: Ising Matrix

    :return: np.ndarray: QUBO Matrix
    :return: float: Offset
    """
    n = J.shape[0]
    qubo = 4*(J - np.diag(np.ones(n).T @ J))
    return qubo, np.sum(J)


def pure_QUBO_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate Ising Matrix J and offset o from Q, such that
    s^T J s + o equals x^T Q x
    with x in {0,1}^n and s in {+- 1}^n,
    n = number of variables
    The transformation x_i = (1+s_i)/2 holds.

    :param Q: np.ndarray: QUBO Matrix

    :return: np.ndarray: Quadratic Ising matrix
    :return: np.ndarray: Linear terms
    :return: float: Offset
    """
    J = 0.25*Q
    np.fill_diagonal(J, 0.)

    h = 0.5 * np.sum(Q, axis=1)
    o = 0.25 * (np.sum(Q) + np.trace(Q))
    return J, h, o


def maxcut_graph_to_ising(G: nx.Graph) -> Tuple[np.ndarray, float]:
    """
    Calculates Ising model from MAXCUT graph

    :param G: nx.Graph: MAXCUT graph

    :return: np.ndarray: Ising matrix
    :return: Ising offset
    """
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    m_ising = 0.25 * adjacency_matrix
    offset = - 0.25 * np.sum(adjacency_matrix)

    return m_ising, offset


def maxcut_graph_to_qubo(G: nx.Graph) -> np.ndarray:
    """
    Calculates QUBO matrix from MAXCUT graph

    :param G: nx.Graph: MAXCUT graph

    :return: np.ndarray: QUBO matrix
    """

    adjacency_matrix = nx.adjacency_matrix(G).todense()
    n = adjacency_matrix.shape[0]

    qubo = adjacency_matrix - np.diag(np.ones(n).T @ adjacency_matrix)

    return qubo


def provide_random_QUBO(nqubits: int, problem_seed: int = 777) \
        -> np.ndarray:
    """
    Generates a randomly created QUBO from uniform distribution

    :param nqubits: int: Number of qubits / nodes in the problem
    :param problem_seed: Seed for numpys default random number generator

    :return: np.ndarray: QUBO matrix
    """
    global problems

    prob_key = f"random_{nqubits}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    a = np.random.default_rng(seed=problem_seed).random((nqubits, nqubits))
    qubo = np.tril(a) + np.tril(a, -1).T
    qubo = (qubo-0.5)*2
    problems[prob_key] = qubo
    return qubo


def provide_random_maxcut_ising(nqubits: int, p: float, problem_seed: int = 777) \
        -> Tuple[np.ndarray, float]:
    """
    Generates random MaxCut Instances from Erdos-Renyi-Graphs
    The resulting graph gets mapped to a Ising matrix and offset.

    :param nqubits: int: Number of nodes (and number of qubits)
    :param p: float: Probability of randomly added edges
    :param problem_seed: int=777: Random seed for networkx graph creation

    :return: np.ndarray: Quadratic Ising matrix
    :return: float: Offset
    """
    global problems

    prob_key = f"ising_MC_{nqubits}_{p}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    g = nx.generators.erdos_renyi_graph(nqubits, p, seed=problem_seed)
    m_ising, offset = maxcut_graph_to_ising(g)

    problems[prob_key] = (m_ising, offset)
    return m_ising, offset


def provide_random_maxcut_QUBO(nqubits: int, p: float, problem_seed: int = 777) \
        -> np.ndarray:
    """
    Generates random MaxCut Instances from Erdos-Renyi-Graphs
    The resulting graph gets mapped to a QUBO.

    :param nqubits: int: Number of nodes (and number of qubits)
    :param p: float: Probability of randomly added edges
    :param problem_seed: int=777: Random seed for networkx graph creation

    :return: np.ndarray: QUBO Matrix
    """
    global problems

    prob_key = f"qubo_MC_{nqubits}_{p}_{problem_seed}"
    if prob_key in problems:
        return problems[prob_key]

    g = nx.generators.erdos_renyi_graph(nqubits, p, seed=problem_seed)
    qubo = maxcut_graph_to_qubo(g)

    problems[prob_key] = qubo
    return qubo


def np_bits2string(a: np.ndarray) -> str:
    """
    Converts a numpy array of bits [0,1] or Ising "bits" [-1,1] to a string
    consisting only of "0" and "1" with completely flattened dimensions.

    :param a: np.ndarray: bit array

    :return: str: bitstring
    """
    a = np.concatenate(np.where(a == -1, 0, 1))
    s = np.array2string(a, separator=" ")
    return s.replace("[", "").replace("]", "").replace("\n", "")


def string2np_bits(s: str, n: int, binary: bool = True) -> Optional[np.ndarray]:
    """
    Converts a string consisting only of "0" and "1" to multidimensional numpy
    array with shape [., n].

    :param s: str: bitstring
    :param n: int: size of subarray
    :param binary: bool: When set to True the resulting bit set consists of
                [0, 1], else of [-1, 1] (Default True)

    :return: np.ndarray: bit array
    """
    bits = None

    for i in range(0, len(s) // 2, n):
        b = np.expand_dims(np.fromstring(s[2*i : 2*(i+n)], sep=" ", dtype="float"), axis=0)
        if bits is None:
            bits = b
        else:
            bits = np.concatenate([bits, b])

    if not binary:
        bits = np.where(bits == 0., -1., 1.)

    return bits


def brute_forced_solution(J: np.ndarray,
                         h: Optional[np.ndarray] = None,
                         offset: float = 0.,
                         binary: bool = True,
                         logfile_name: str = "brute_forced_solutions.csv",
                         generation_seed: Optional[int] = None,
                         ) \
        -> Tuple[Optional[np.ndarray], float]:
    """
    Brute forces best solution for a quadratic minimisation problem.
    If multiple assignments correspond to the same minimum cost function,
    all of them get returned.
    If the generation seed is provided solutions are loaded in a logfile if the
    they are already computed, else they get stored.

    :param J: np.ndarray: QUBO or Ising matrix
    :param h: Optional(np.ndarray): Linear terms for Ising model (Default None)
    :param offset: float: Ising model offset (Default 0.)
    :param binary: bool: If set to True the variable assignment is binary
                (in [0,1], QUBO) else the in [-1, 1] (Ising) (Default False)
    :param logfile_name: str: name of the logfile, where the brute forced solutions
                get stored
    :param generation_seed: Optional(int): The seed with which the problem was
                generated (Default None)

    :return: Optional(np.ndarray): List of variable assignment(s) for the optimal
                solution(s)
    :return: float: Optimal cost value
    """
    n = J.shape[0]
    idx = 0

    if generation_seed and os.path.exists(logfile_name):
        df = pd.read_csv(logfile_name)
        idx = df.shape[0]
        data = df.loc[(df['seed'] == generation_seed) & (df["n"] == n)]
        if data.shape[0] > 0:
            cost = data["min_cost"].tolist()[0]
            bit_str = data["var_assignment"].tolist()[0]
            bits = string2np_bits(bit_str, n, binary)
            return bits, cost

    bits = [0, 1] if binary else [-1, 1]
    min_cost = np.inf
    min_bits = None

    for x in product(bits, repeat=n):
        x = np.array(x)
        cost = cost_function(J, x, h, offset)

        if cost < min_cost or min_bits is None:
            min_cost = cost
            min_bits = np.expand_dims(x, axis=0)
        elif cost == min_cost:
            min_bits = np.concatenate([min_bits, np.expand_dims(x, axis=0)])

    if generation_seed:
        bit_str = np_bits2string(min_bits)
        data = {
                "n": n,
                "seed": generation_seed,
                "min_cost": min_cost,
                "var_assignment": bit_str,
                }

        df = pd.DataFrame(data, index = [idx])
        if os.path.exists(logfile_name):
            df.to_csv(logfile_name, mode="a", header=False)
        else:
            df.to_csv(logfile_name)

    return min_bits, min_cost


def cost_function(J: np.ndarray,
                  x: np.ndarray,
                  h: Optional[np.ndarray] = None,
                  offset: float = 0.,
                  ) \
        -> np.ndarray:
    """
    Computes x^T * J * x + hT * x + o for a given bitstring using numpy operations

    :param J: np.ndarray: QUBO or Ising matrix
    :param x: np.ndarray: Variable assignment either in [0,1] (QUBO)
              or in [-1, 1] (Ising)
    :param h: Optional(np.ndarray): Linear terms for Ising model (Default None)
    :param offset: float: Ising model offset (Default 0)

    :return: float: cost for variable assignment
    """
    cost = x.T @ J @ x

    if h is not None:
        cost += h.T @ x

    cost += offset

    return cost


def cost_function_batched(J: tf.Tensor,
                  x: tf.Tensor,
                  h: Optional[tf.Tensor] = None,
                  offset: Optional[tf.Tensor] = None,
                  ) \
        -> tf.Tensor:
    """
    Computes x^T * J * x + hT * x + o for a given batch using tensorflow operations

    :param J: tf.Tensor: QUBO or Ising matrix
    :param x: tf.Tensor: Variable assignment either in [0,1] (QUBO)
              or in [-1, 1] (Ising)
    :param h: Optional(tf.Tensor): Linear terms for Ising model (Default None)
    :param offset: Optional(tf.Tensor): Ising model offset (Default None)

    :return: tf.Tensor: Batched costs for variable assignments
    """
    Jx = tf.einsum("bij,bj->bi", J, x)
    cost = tf.einsum("bi,bi->b", x, Jx)

    if h is not None:
        cost += tf.einsum("bi,bi->b", x, h)

    if offset is not None:
        cost += offset

    return cost

