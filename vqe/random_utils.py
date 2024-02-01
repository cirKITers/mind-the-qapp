import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed: int = 27) -> None:
    """
    Sets the global tensorflow, numpy and random seed.
    Pennylane uses the numpy seed for sampling.

    :param seed: int: Global seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
