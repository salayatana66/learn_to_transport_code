"""
Samplers for the distributions \mu & \nu
\mu is uniform on the unit ball in R^2
\nu is made of 4 balls in R^2
"""
import numpy as np

from typing import Tuple, Optional


def circle_sampler_loc(size: int) -> Tuple[np.array, np.array]:
    """
    Sample uniformly in the unit ball in R^2
    using U[0,1] \times U[0,1]
    and rejection sampling

    :param size: sample size for U[0,1] x U[0,1]
    :return sample coordinates
    """
    x = np.random.uniform(low=-1.0, high=1.0, size=size)
    y = np.random.uniform(low=-1.0, high=1.0, size=size)
    msk = (x * x + y * y <= 1.0)

    return x[msk], y[msk]
                    

def circle_sampler(size: int) -> Tuple[np.array, np.array]:
    """
    Yield a sample of size size on the unit ball in R^2
    applying the sampler_loc repeatedly
    :param size: required size
    :return: sample coordinates
    """

    out_len = 0
    x: Optional[np.array] = None
    y: Optional[np.array] = None

    while out_len < size:
        x_, y_ = circle_sampler_loc(size)

        if x is None:
            x = x_
            y = y_
        else:
            x = np.concatenate((x, x_))
            y = np.concatenate((y, y_))

        out_len = len(x)

    return x[:size], y[:size]


def lobate_circles(size: int) -> Tuple[np.array, np.array]:
    """
    Sample the \nu distribution
    :param size: sample size
    :return: sample coordinates
    """

    radius = .5

    centers = np.array([[1.0, 1.0],
                        [-1.0, 1.0],
                        [1.0, -1.0],
                        [-1.0, -1.0]])

    # sample centers
    csel = np.random.choice(centers.shape[0],
                            size=size)

    x_, y_ = circle_sampler(size)

    # sample around the centers
    x_out = centers[csel, 0] + radius * x_
    y_out = centers[csel, 1] + radius * y_

    return x_out, y_out

