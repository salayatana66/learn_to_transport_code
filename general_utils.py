"""
General utilities used across the scripts

@author Andrea Schioppa
"""

import matplotlib.pyplot as plt
import os
import shutil
import logging

from pycrayon import CrayonClient
from pycrayon.crayon import CrayonExperiment
from torch import Tensor
from torch.optim import Optimizer
from typing import Dict, List


def get_logger(logger_name: str) -> logging.Logger:
    """
    Default logger, put in INFO mode
    """

    logger = logging.getLogger(logger_name)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s @%(lineno)s - %(name)s - %(levelname)s : %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    return logger


def create_directory(dir_name: str, delete_if_exists: bool = False) -> str:
    """
    Create directory in safe way
    :param dir_name: directory name (user variables will get expanded)
    :param delete_if_exists: if the directory exists do not fail, but delete content
    :return: directory name with user / shell variables expanded
    """

    dir_path = (os.path.expanduser(
        os.path.expandvars(dir_name)))

    try:
        os.makedirs(dir_path)
        
    except FileExistsError as ferr:
        if not delete_if_exists:
            raise ferr
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)   

    return dir_path


def crayon_create_experiment(exp_name: str, cclient: CrayonClient,
                             overwrite: bool = True) -> CrayonExperiment:
    """
    Create experiment name in the alband/crayon tensorboard
    :param exp_name: name of experiment
    :param cclient: handler of requests to crayon
    :param overwrite: if the experiment already exists delete and recreate
    :return:
    """
    
    try:
        ccexp = cclient.create_experiment(exp_name)
        return ccexp

    except ValueError as verr:
        if overwrite:

            cclient.remove_experiment(exp_name)
            ccexp = cclient.create_experiment(exp_name)
            return ccexp
        else:

            raise verr


def visualize(X: Tensor, Y: Tensor,
              TX: Tensor, fname: str) -> None:
    """
    visualize the transport map X -> Y via TX
    generates a .png
    :param X: X ~ \mu
    :param Y: Y ~ \nu
    :param TX: T(X)
    :param fname: name of output file
    """

    plt.figure()

    # Plot X, Y, TX
    plt.plot(X.detach().numpy()[:, 0],
             X.detach().numpy()[:, 1], '.')
    plt.plot(Y.detach().numpy()[:, 0],
             Y.detach().numpy()[:, 1], 'k.')
    plt.plot(TX.detach().numpy()[:, 0],
             TX.detach().numpy()[:, 1], 'r.')

    # Plot lines to simulate movement
    for j in range(X.shape[0]):
        plt.arrow(X.detach().numpy()[j, 0],
                  X.detach().numpy()[j, 1],
                  TX.detach().numpy()[j, 0] - X.detach().numpy()[j, 0],
                  TX.detach().numpy()[j, 1] - X.detach().numpy()[j, 1])
        
    # save; close needed to save memory effort
    plt.savefig(fname)
    plt.close()


def visualize_w_optimal(X: Tensor, Y: Tensor,
                        TX: Tensor, Topt: Tensor, fname: str):
    """
    variant of visualize where TX can be compared with a better
    (even the optimal) Topt
    """

    plt.figure()
    # Plot X, Y, TX
    plt.plot(X.detach().numpy()[:, 0],
             X.detach().numpy()[:, 1], '.')
    plt.plot(Y.detach().numpy()[:, 0],
             Y.detach().numpy()[:, 1], 'k.')
    plt.plot(TX.detach().numpy()[:, 0],
             TX.detach().numpy()[:, 1], 'r.')

    # Plot lines to simulate movement
    for j in range(X.shape[0]):
        plt.arrow(X.detach().numpy()[j, 0],
                  X.detach().numpy()[j, 1],
                  TX.detach().numpy()[j, 0] - X.detach().numpy()[j, 0],
                  TX.detach().numpy()[j, 1] - X.detach().numpy()[j, 1])
        
    # save; close needed to save memory effort
    # Plot lines to simulate movement
    for j in range(X.shape[0]):
        plt.arrow(X.detach().numpy()[j, 0],
                  X.detach().numpy()[j, 1],
                  Topt[j, 0] - X.detach().numpy()[j, 0],
                  Topt[j, 1] - X.detach().numpy()[j, 1])
    
    plt.savefig(fname)
    plt.close()


def torch_losses_take_step(loss_dict: Dict[str, Tensor],
                           optimizer: Optimizer, loss_names: List[str],
                           minimize: bool = True,
                           retain_graph: bool = False) -> None:
    """
    Take one SGD step
    :param loss_dict: dictionary of losses
    :param optimizer: optimizer to use (e.g. for tuning step size / gradient direction)
    :param loss_names: names of the losses to aggregate
    :param minimize: are we minimizing or maximizing ?
    :param retain_graph: if we plan to take further steps before a forward propagation,
       we need to retain the graph
    """

    optimizer.zero_grad()
    
    # sum needed losses
    sum_losses = sum(v for k, v in loss_dict.items() if
                     k in loss_names)
    if not minimize:
        sum_losses = -1.0 * sum_losses

    sum_losses.backward(retain_graph=retain_graph)
    optimizer.step()


def roll_average(loss_dict: Dict[str, Tensor], mets_dict: Dict[str, float],
                 metrics: List[str], iteration: int) -> None:
    """
    Update rolling averages
    :param loss_dict: Torch tensors
    :param mets_dict: dictionary holding current rolling averages
    :param metrics: list of metrics to update
    :param iteration: current iteration
    """

    for met in metrics:
        mets_dict[met] += (float(loss_dict[met].detach().numpy())
                           - mets_dict[met]) / (iteration + 1)


def crayon_ship_metrics(crayon_exp: CrayonExperiment,
                        mets_dict: Dict[str, float],
                        metrics: List[str],
                        iteration: int):
    """
    Ship new metrics to alband/crayon's tensorboard
    :param crayon_exp: target experiment
    :param mets_dict: dictionary holding metrics
    :param metrics: list of metrics to ship
    :param iteration: current iteration
    :return:
    """

    out_dict = dict([(k, v) for k, v in mets_dict.items()
                     if k in metrics])
    crayon_exp.add_scalar_dict(out_dict, wall_time=-1, step=iteration + 1)

