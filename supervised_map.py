"""
Learning a transport map
using Sinkhorn Iterations to learn
the map directly

@author: Andrea Schioppa
"""


import numpy as np
import argparse
import datetime
import torch

# ! with Sinkhorn things it is advisable to
# increase precision
torch.set_default_dtype(torch.float64)

from torch.utils.data import DataLoader
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.expanduser("~/learn_to_transport_code"))

from general_utils import crayon_create_experiment
from general_utils import create_directory
from general_utils import crayon_ship_metrics
from general_utils import torch_losses_take_step
from general_utils import roll_average
from general_utils import get_logger

# The Dataset
from circ_distros import CircDistros

# The distance metrics
from prob_dist_meas import square_distances as square_distances_fn

# note potentials are 1-dimensional so that's why we use _simple distance metric
from prob_dist_meas import identity_loss as identity_loss_fn
from prob_dist_meas import l2_loss as l2_loss_fn

# The transporter
from estimators import NeuralTransportMap

from pycrayon import CrayonClient

from typing import List, Tuple

from common_evaluation import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str, default=None)
parser.add_argument("--base_plots_dir", type=str, default="~/opt_w_dl_plots_1")
parser.add_argument("--base_models_dir", type=str, default="~/opt_w_dl_models_1")
parser.add_argument("--base_evaluation_dir", type=str, default="~/opt_w_dl_evaluation_1")
parser.add_argument("--numpy_seed", type=int, default=15)
parser.add_argument("--n_init", type=int, required=True,
                    help="Iterations to initialize transport map & the potentials")
parser.add_argument("--n_train", type=int, required=True)
parser.add_argument("--b_final_eval", type=int, default=1000, help="Batch size for final evaluation")
parser.add_argument("--epsilon", type=float, required=True,
                    help="epsilon for regularization")
parser.add_argument("--max_inner_iter", type=int, required=True,
                    help="Maximal # of iterations in the inner supervised loop")
parser.add_argument("--inner_sink_iter", type=int, required=True,
                    help="# of Sinkhorn iterations")
parser.add_argument("--crayon_send_stats_iters", type=int, default=20,
                    help="How many iters we send stats to tensorboard")
parser.add_argument("--n_models_saved", type=int, default=50,
                    help="How many snapshot of intermediate models we save")
args = parser.parse_args()
logger = get_logger("supervised_map")

if args.today is None:
    args.today = datetime.date.today().strftime("%Y-%m-%d")

logger.info(f"Using experiment date: {args.today}")

# Experiment Name
exp_name = f"supervised_map_{args.n_init}_ntrain_{args.n_train}_eps_{args.epsilon}_mxinneriter_{args.max_inner_iter}_innsink_{args.inner_sink_iter}_{args.today}"
logger.info(f"Experiment name: {exp_name}")

# create directories for plots, models & evaluation
plots_dir = f"{args.base_plots_dir}/{exp_name}"
logger.info(f"For plotting using dir: {plots_dir}")
plots_dir = create_directory(plots_dir, delete_if_exists=True)

save_dir = f"{args.base_models_dir}/{exp_name}"
logger.info(f"For model saving using dir: {save_dir}")
save_dir = create_directory(save_dir, delete_if_exists=True)

export_dir = f"{args.base_evaluation_dir}/{exp_name}"
logger.info(f"For exporting final evaluation using dir: {export_dir}")
export_dir = create_directory(export_dir, delete_if_exists=True)

# Connect to server & start experiment
ccexp = crayon_create_experiment(exp_name, CrayonClient())

# seed
logger.info(f"Using seed: {args.numpy_seed}")
np.random.seed(args.numpy_seed)

# metrics to send to tensorboard
mets = {"identity_loss": 0.0,
        "marginal_constraint_u": 0.0,
        "marginal_constraint_v": 0.0,
        "l2_loss_tm": 0.0
        }

# initialization & training iterations
N_init = args.n_init
logger.info(f"Will train initialization for {N_init} iterations")

N_train = args.n_train
logger.info(f"Will train transport map for {N_train} iterations")

max_inner_iter = args.max_inner_iter
logger.info(f"Each supervised training loop will last for at most {max_inner_iter} iterations")

inner_sink_iter = args.inner_sink_iter
logger.info(f"Each invocation of Sinkhorn algorithm will use {inner_sink_iter} iterations")

B_final_eval = args.b_final_eval
logger.info(f"Final batch size for evaluations & movie will be {B_final_eval}")

epsilon = args.epsilon
logger.info(f"Regularization strength will be {epsilon}")

# initialize the samplers
crc_dist_burnin = CircDistros(size=N_init, sample_size=256)
crc_dist = CircDistros(size=N_train, sample_size=256)
crc_final = CircDistros(size=1, sample_size=B_final_eval)

# initialize the dataloading for PyTorch
dataloader_init = DataLoader(dataset=crc_dist_burnin, batch_size=1,
                             shuffle=True, drop_last=True)
dataloader_train = DataLoader(dataset=crc_dist, batch_size=1,
                              shuffle=True, drop_last=True)
dataloader_final_eval = DataLoader(dataset=crc_final, batch_size=1,
                                   shuffle=True, drop_last=True)

#####################################
# The models & the optimizers
#####################################
neural_map = NeuralTransportMap(space_dim=2, layers_dim=[128, 64])
opt_tm = torch.optim.Adagrad(neural_map.parameters(), lr=1e-2)

# how many times we send stats to tensorboard
n_stats_to_tensorboard = args.crayon_send_stats_iters
logger.info(f"Sending stats to tensorboard every {n_stats_to_tensorboard} iterations")

# how many times we save model
n_save: int = round(args.n_train / args.n_models_saved)
logger.info(f"Save models every {n_save} iterations, for a total of {args.n_models_saved}")


def sink_iterate_find_potentials(batchSize: int, C: Tensor,
                                 Niter: int, sink_reg: float = 1e-2)\
        -> Tuple[Tensor, Tensor]:
    """
    Plain Sinkhorn in Pytorch, assuming u & v have same dimensionality
    :param batchSize: dimension of u, v
    :param C: costs
    :param Niter: number of iterations
    :param sink_reg: sinkhorn regularization
    :return:
    """

    # initialize the marginals 
    a = torch.ones(batchSize) * 1.0/batchSize
    b = torch.ones(batchSize) * 1.0/batchSize

    # Gibbs matrix
    K = torch.exp(- C / sink_reg)

    # initial condition w/o backprop
    v = b.detach()
    u = a.detach()
    for _ in range(Niter):
        u = a / torch.matmul(K, v)
        v = b / torch.matmul(K.t(), u)

    return u, v


def sink_comp_constraint(u: Tensor, v: Tensor, C: Tensor,
                         sink_reg: float = 1e-2) -> Tuple[Tensor, Tensor]:
    """
    Compute constraint violation induced by the logs f, g of
    potentials
    :param u: solution to X-potential
    :param v: solution to Y-potential
    :param C: cost
    :param sink_reg: regularization
    """

    # Gibbs matrix
    K = torch.exp(- C / sink_reg)

    onU = u.view(-1) * torch.matmul(K, v).view(-1)
    onV = v.view(-1) * torch.matmul(K.t(), u).view(-1)

    return onU, onV


def sink_comp_find_map(u0: Tensor, v0: Tensor,
                       C: Tensor, Y: Tensor, sink_reg: float = 1e-2)\
        -> Tensor:
    """
    Given a solution to Sinkhorn's dual problem, compute the image
    of the X ~\mu sample under the transport map (using essentially
    the fiberwise heuristic on Y~\nu)
    :param u0: solution to dual
    :param v0: solution to dual
    :param C: cost
    :param Y: sample from \nu
    :param sink_reg: regularization strength
    """
    # Gibbs matrix
    K = torch.exp(- C / sink_reg)

    # initial condition w/o backprop
    u = u0.clone().detach()
    v = v0.clone().detach()

    # compute probability of association
    pi = u.view((-1, 1)) * K * v.view((1,-1))

    # rescale to a probability
    pi = pi / torch.sum(pi, dim=1).view((-1, 1))

    # compute heuristically the targets
    Yavg = torch.matmul(pi, Y).detach()

    return Yavg


# initialize network to the identity
for iteration, data_dict in enumerate(dataloader_init):
    X = data_dict["X"].squeeze(dim=0)
    TX = neural_map(X)

    torch_losses = {
        "identity_loss": identity_loss_fn(X, TX)
    }

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_tm,
                           loss_names=["identity_loss"])

    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["identity_loss"],
                 iteration=iteration)
    if (iteration + 1) % n_stats_to_tensorboard:
        crayon_ship_metrics(ccexp, mets, ["identity_loss"],
                            iteration)

# iterations to evaluate on
eval_iters: List[int] = []

# supervised training loop
# keeps track of the current step
step_counter = 0

for iteration, data_dict in enumerate(dataloader_train):

    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)

    # compute cost
    Cdist = square_distances_fn(X, Y)

    # compute batch size
    batchSize = X.shape[0]

    # solve Sinkhorn's
    u, v = sink_iterate_find_potentials(batchSize, Cdist,
                                        inner_sink_iter, epsilon)
    # constraints pertaining to convergence
    onU, onV = sink_comp_constraint(u, v, Cdist, epsilon)

    # where we want to map X to
    Yavg = sink_comp_find_map(u, v, Cdist, Y, epsilon)

    marginal_constraint_u = (onU - 1/batchSize).abs().mean()
    marginal_constraint_v = (onV - 1/batchSize).abs().mean()

    local_counter = 0
    while local_counter < max_inner_iter:
        # fit the loss
        TX = neural_map(X)

        torch_losses = {
            # Using sum for l2_loss speeds up convergence
            "l2_loss_tm": ((TX-Yavg)*(TX-Yavg)).sum(),
            "marginal_constraint_u": marginal_constraint_u,
            "marginal_constraint_v": marginal_constraint_v
        }


        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_tm,
                               loss_names=["l2_loss_tm"],
                               retain_graph=True)

        roll_average(loss_dict=torch_losses, mets_dict=mets,
                     metrics=["l2_loss_tm", "marginal_constraint_u",
                              "marginal_constraint_v"],
                     iteration=step_counter)

        if (step_counter + 1) % n_stats_to_tensorboard == 0:
            crayon_ship_metrics(ccexp, mets, ["l2_loss_tm", "marginal_constraint_u",
                                              "marginal_constraint_v"],
                                step_counter)

        if (step_counter + 1) % n_save == 0:
            torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_{step_counter}.model")
            eval_iters.append(step_counter)

        local_counter += 1
        step_counter += 1

    if step_counter > N_train:
        break

evaluate(eval_iters, neural_map, save_dir, export_dir, plots_dir,
         crc_final)






