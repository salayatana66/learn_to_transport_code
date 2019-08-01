"""
Sinkhorn iterations (suboptimal loop code in computing
distances)

@author: Andrea Schioppa
"""

import numpy as np

from itertools import product
from typing import Optional, Tuple


def compute_pairwise_euclidean(x: np.array, y: np.array) -> np.array:
    """
    Compute pairwise l2-distances (squared) between x, y
    :param x: with shape (n, D) -> n # of points & D dimensions of ambient space
    :param y: with shape (m, D)
    :return: with shape (n, m)
    """

    n = x.shape[0]
    m = y.shape[0]

    if x.shape[1] != y.shape[1]:
        raise ValueError(""" x & y represent points in Euclidean
                             spaces of different dimensions
        """)

    C = np.zeros((n, m))
    for i, j in product(range(n), range(m)):
        C[i, j] = np.linalg.norm(x[i, :] - y[j, :])**2

    return C


def sinkhorn_iterations(n_iter: int, a: np.array,
                        b: np.array, C: np.array, epsilon: float,
                        v: Optional[np.array] = None) -> Tuple[np.array, np.array, np.array]:
    """
    Performs Sinkhorn iterations in the potential space (no logarithmic regularization)
    :param n_iter: iterations to do
    :param a: target marginal in X space
    :param b: target marginal in Y space
    :param C: Cost with shape (#X, #Y)
    :param epsilon: regularization epsilon
    :param v: if supplied an initial v, otherwise defaults to np.ones_like(b)
    :return: (unnormalized) plan P, constraint violation on X, constraint violation on Y
    """

    if v is None:
        v = np.ones_like(b, dtype=np.float64)

    # Gibbs cost
    K = np.exp(- C / epsilon)

    # constraint-enforcing loop
    for i in range(n_iter):
        u = a/K.dot(v)
        v = b/K.T.dot(u)

    # copmuting final constraint violations
    viol_a = np.sum(np.abs(
        (u * K.dot(v) - a).flatten()))
    viol_b = np.sum(np.abs(
        (v * K.T.dot(u) - b).flatten()))

    # final distribution
    P = np.diag(u).dot(K.dot(np.diag(v)))

    return P, viol_a, viol_b
