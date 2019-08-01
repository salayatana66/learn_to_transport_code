"""
Representations of transport maps,
plans, potentials and adversarial constraints

@author: Andrea Schioppa
"""


import torch
import torch.nn as nn
from collections import OrderedDict

from typing import List


class NeuralTransportMap(nn.Module):
    """
    Transport Map (or component of a transport map)
    from R^D -> R^D
    parametrized via a neural network
    """

    def __init__(self, space_dim: int, layers_dim: List[int]):
        """
        :param space_dim: dimension of ambient space
        :param layers_dim: layers dimensions
        """

        super(NeuralTransportMap, self).__init__()

        layers_ = []
        self.space_dim = space_dim
        self.layers_dim = layers_dim

        layers_.append(("base_layer", nn.Linear(in_features=self.space_dim,
                                                out_features=self.layers_dim[0])))

        # we use hardtanh to constrain intermediate features between -1,1
        for i, d in enumerate(layers_dim[:-1]):
            layers_.append(("hardtanh" + str(i), nn.Hardtanh()))
            layers_.append(("fc" + str(i + 1), nn.Linear(
                in_features=d, out_features=layers_dim[i + 1])))

        # final output layer
        layers_.append(("hardtanh" + str(i + 1), nn.Hardtanh()))
        layers_.append(("output", nn.Linear(in_features=layers_dim[-1],
                                            out_features=self.space_dim)))

        self.model = nn.Sequential(OrderedDict(layers_))
        self.add_module("neural_network", self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class NeuralTransportPlan(nn.Module):
    """
    Transport Plan Probability Evauation
    it is a map R^D X R^D -> [0,\infty)
    parametrized via a neural network
    Note: no rescaling to probability is enforced
    """

    def __init__(self, space_dim: int, layers_dim: List[int]):
        """
        :param space_dim: dimension of ambient space
        :param layers_dim: layers dimensions
        """

        super(NeuralTransportPlan, self).__init__()

        layers_ = []
        self.space_dim = space_dim
        self.layers_dim = layers_dim

        layers_.append(("base_layer", nn.Linear(in_features=self.space_dim * 2,
                                                out_features=self.layers_dim[0])))

        # we use hardtanh to constrain intermediate features between -1,1
        for i, d in enumerate(layers_dim[:-1]):
            layers_.append(("hardtanh" + str(i), nn.Hardtanh()))
            layers_.append(("fc" + str(i + 1), nn.Linear(
                in_features=d, out_features=layers_dim[i + 1])))

        # final output layer
        layers_.append(("hardtanh" + str(i + 1), nn.Hardtanh()))
        layers_.append(("output_score", nn.Linear(in_features=layers_dim[-1],
                                            out_features=1)))
        layers_.append(("output_sigmoid", nn.Sigmoid()))

        self.model = nn.Sequential(OrderedDict(layers_))
        self.add_module("neural_network", self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class IdentityPerturbation(nn.Module):
    """
    Convex perturbation of the identity
    rho * Id + (1-rho) * tangent
    """

    def __init__(self, rho: float, tangent: nn.Module):
        """
        @param: rho (double): part attributed to the identity
        @param tangent: the module that perturbs the identity
        """

        super(IdentityPerturbation, self).__init__()
        self.add_module("tangent", tangent)
        self.rho_tensor = nn.Parameter(torch.tensor(rho, requires_grad=True))
        self.register_parameter("rho", self.rho_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (x + (self.rho_tensor
                     * self._modules["tangent"](x)))


class TransportPlan(nn.Module):
    r"""
    Transport Plan as a generative model on
    Z --> [-target_box, target_box]^D \times [-target_box, target_box]^D
    parametrized via a neural network
    Note: we don't force the mass to be 1
    """

    def __init__(self, space_dim: int,
                 latent_dim: int, target_box: float, layers_dim: List[int]):
        """
        :param space_dim: dimension of target space
        :param latent_dim: dimension of the random generator feeding the network
        :param target_box: > 0, generated samples are bound in [-target_box, target_box]^D
        :param layers_dim: dimensions of intermediate layers of the neural network
        """

        super(TransportPlan, self).__init__()

        layers_ = []
        self.space_dim = space_dim
        self.latent_dim = latent_dim
        self.layers_dim = layers_dim
        self.target_box = target_box

        layers_.append(("base_layer", nn.Linear(in_features=self.latent_dim,
                                                out_features=self.layers_dim[0])))

        # we use hardtanh to constrain intermediate features between -1,1
        for i, d in enumerate(layers_dim[:-1]):
            layers_.append(("hardtanh" + str(i), nn.Hardtanh()))
            layers_.append(("fc" + str(i + 1), nn.Linear(
                in_features=d, out_features=layers_dim[i + 1])))

        # final output layer
        layers_.append(("hardtanh" + str(i + 1), nn.Hardtanh()))
        self.gen_model = nn.Sequential(OrderedDict(layers_))

        self.fcx = nn.Linear(in_features=layers_dim[-1],
                             out_features=self.space_dim)
        self.fcy = nn.Linear(in_features=layers_dim[-1],
                             out_features=self.space_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        final_layer = self.gen_model(z)
        return self.fcx(final_layer), self.fcy(final_layer)


class BoundedFunction(nn.Module):
    r"""
    Bounded Continuous Function R^D -> [-1,1]
    parametrized via a neural network
    """

    def __init__(self, space_dim: int, layers_dim: List[int]):
        """
        :param space_dim: dimension of ambient space
        :param layers_dim: intermediate dimension of layers
        """

        super(BoundedFunction, self).__init__()

        layers_ = []
        self.space_dim = space_dim
        self.layers_dim = layers_dim

        layers_.append(("base_layer", nn.Linear(in_features=self.space_dim,
                                                out_features=self.layers_dim[0])))

        # we use hardtanh to constrain intermediate features between -1,1
        for i, d in enumerate(layers_dim[:-1]):
            layers_.append(("relu" + str(i), nn.ReLU()))
            layers_.append(("fc" + str(i + 1), nn.Linear(
                in_features=d, out_features=layers_dim[i + 1])))

        # final output layer
        layers_.append(("relu" + str(i + 1), nn.ReLU()))
        layers_.append(("fc" + str(i + 2), nn.Linear(in_features=layers_dim[-1],
                                                     out_features=1)))
        layers_.append(("output", nn.Hardtanh()))

        self.model = nn.Sequential(OrderedDict(layers_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        !! output is bounded in [-1,1]
        """

        return self.model(x)


class PotentialFunction(nn.Module):
    r"""
    Like Bounded Continuous Function R^D -> R;
    we remove the final arctanh
    """

    def __init__(self, space_dim: int, layers_dim: List[int]):
        """
        :param space_dim: dimension of the ambient space
        :param layers_dim: dimension of intermediate layers of the neural network
        """

        super(PotentialFunction, self).__init__()

        layers_ = []
        self.space_dim = space_dim
        self.layers_dim = layers_dim

        layers_.append(("base_layer", nn.Linear(in_features=self.space_dim,
                                                out_features=self.layers_dim[0])))

        # we use hardtanh to constrain intermediate features between -1,1
        for i, d in enumerate(layers_dim[:-1]):
            layers_.append(("relu" + str(i), nn.ReLU()))
            layers_.append(("fc" + str(i + 1), nn.Linear(
                in_features=d, out_features=layers_dim[i + 1])))

        # final output layer
        layers_.append(("relu" + str(i + 1), nn.ReLU()))
        layers_.append(("fc" + str(i + 2), nn.Linear(in_features=layers_dim[-1],
                                                     out_features=1)))

        self.model = nn.Sequential(OrderedDict(layers_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x).view(-1, )
