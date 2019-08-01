"""
Generated points for the lobate circle distribution

@author: Andrea Schioppa
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from circle_samplers import circle_sampler
from circle_samplers import lobate_circles

import sys
import os
sys.path.insert(0, os.path.expanduser("~/learn_to_transport_code"))


class CircDistros(Dataset):
    """
    Our \mu & \nu
    """

    def __init__(self, size: int, sample_size: int):
        """
        :param size: maximum number of samples we can generate
               in total, counting across batches
        :param sample_size: number of points to generate for each batch
        """

        self.size = size
        self.sample_size = sample_size

    def __len__(self):

        return self.size

    def __getitem__(self, index):

        xuf, yuf = circle_sampler(self.sample_size)
        xlob, ylob = lobate_circles(self.sample_size)

        cuf = np.stack((xuf, yuf), axis=1)
        clb = np.stack((xlob, ylob), axis=1)

        return {"X": torch.Tensor(cuf), "Y": torch.Tensor(clb)}

