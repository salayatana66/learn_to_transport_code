"""
Different distances & losses we use in
PyTorch experiments
@author Andrea Schioppa
"""

import torch

from typing import Callable

def covariance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes empirical covariance matrics between X & Y
    :param X: with shape (B, D1)
    :param Y: with shape (B, D2)
    :return: empirical covariance matrix with shape (D1, D2)
    """

    # means 
    mX = torch.mean(X, dim=0)
    mY = torch.mean(Y, dim=0)

    # centered
    cX = X - mX
    cY = Y - mY

    return torch.matmul(cX.t(), cY) / cX.shape[0]


def square_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise square distances between rows of X & Y
    :param X: with shape (B1, D)
    :param Y: with shape (B2, D)
    :return: with shape (B1, B2)
    """

    # extract dimensions
    B1, D1 = X.shape
    B2, D2 = Y.shape
    assert(D1 == D2)
    D = D1
    
    # use casting to compute pairwise differences
    diff = X.view((B1, 1, D)) - Y.view((1, B2, D))

    squared_distances = torch.sum(diff.pow(2), dim=2)

    return squared_distances


def mean_discrepancy(X: torch.Tensor, Y: torch.Tensor, cutoff: int) -> torch.Tensor:
    """
    For each row in X, averaged square cost to the closest cutoff points
    :param X: with shape (B1, D)
    :param Y: with shape (B2, D)
    :param cutoff: integer > 0
    :return: with shape B1
    """

    # all pairwise squared distances
    sqdists = square_distances(X, Y)

    # take cutoff smallest distances
    topped_values, topped_idx = torch\
        .topk(sqdists, k=cutoff, dim=1,
              largest=False)

    # average over closest points
    return topped_values.mean()


def exponential_integrals(X: torch.Tensor, Z: torch.Tensor, sig: float) -> torch.Tensor:
    """
    montecarlo integrals of exp(-|X-Z|^2/sig)
    :param X: with shape (B, D)
    :param Z: with shape (N, D)
    :param sig: with shape (N, )
    :return: (N,) -> MonteCarlo integral for each center
    """

    # compute scaled square distances
    dmat = square_distances(Z, X) / sig.view(Z.shape[0], 1)

    # exponentiate and reduce
    expDiff = torch.mean(torch.exp(- dmat), dim=1)

    return expDiff


def exp_discrepancy(X: torch.Tensor, Y: torch.Tensor,
                    Z: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """
    mean of discrepancies across the exponential integrals
    :param X: sample X ~ \mu
    :param Y: sample Y ~ \nu
    :param Z: centers
    :param sig: dilations in exponential
    :return: total loss
    """

    xint = exponential_integrals(X, Z, sig)
    yint = exponential_integrals(Y, Z, sig)

    # reduce
    return (xint - yint).abs().mean()
    

def identity_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Loss to force X & Y to agree
    :return: |X-Y|^2
    """
   
    return ((X - Y) * (X - Y)).sum()


def means_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Loss between the means of X & Y
    :return: |\mu_X - \mu_Y|^2
    """
    
    mX = torch.mean(X, dim=0)
    mY = torch.mean(Y, dim=0)

    return ((mX - mY) * (mX - mY)).sum()


def covariance_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Loss between the covariance matrices of X & Y
    :return: |cov_X - cov_Y|^2
    """
    
    cVX = covariance(X, X)
    cVY = covariance(Y, Y)

    return ((cVX - cVY) * (cVX - cVY)).sum()


def l2_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    The difference with identity loss is that this sums only across
    the dimension 1, and takes the mean over the batch
    :param X: with shape (B, D)
    :param Y: with shape (B, D)
    :return: 1/B\sum_i||X_i - Y_i||^2
    """

    diff = (X - Y) * (X - Y)

    # reduce across dimension 1
    diff_sum = torch.sum(diff, dim=1)

    # return mean on the batch
    return diff_sum.mean()


def l2_loss_simple_vectors(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    variant of l2_loss for 1-dimensional or 0 dimensional arguments
    """

    diff = (X - Y) * (X - Y)

    # return mean on the batch
    return diff.mean()


def critic_loss(X: torch.Tensor, Y: torch.Tensor,
                C: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    MonteCarlo evaluation of critic loss
    E[C(X)] - E[C(Y)]
    :param X: X ~ \mu
    :param Y: Y ~ \nu
    :param C: The critic/adversary that assigns a value to each sampler
    :return: Montecarlo approximation of E[C(X)] - E[C(Y)]
    """

    Xcritic = C(X)
    Ycritic = C(Y)

    return C(X).mean() - C(Y).mean()
