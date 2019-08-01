"""
Learning a transport map
using Sinkhorn Iterations to learn
potentials in a supervised way

@author: Andrea Schioppa
"""


import numpy as np
import argparse
import datetime
import torch

# ! with Sinkhorn things it is advisable to
# add precision
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
from prob_dist_meas import l2_loss_simple_vectors as l2_loss_simple_vectors_fn
from prob_dist_meas import identity_loss as identity_loss_fn
from prob_dist_meas import l2_loss as l2_loss_fn

# The transporter
from estimators import NeuralTransportMap
# The potential function
from estimators import PotentialFunction

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
parser.add_argument("--n_train_tmap", type=int, required=True,
                    help="Training iterations for the transport map")
parser.add_argument("--b_final_eval", type=int, default=1000, help="Batch size for final evaluation")
parser.add_argument("--epsilon", type=float, required=True,
                    help="epsilon for regularization")
parser.add_argument("--max_inner_iter", type=int, required=True,
                    help="Maximal # of iterations in the inner supervised loop")
parser.add_argument("--max_inner_error", type=float, required=True,
                    help="If supervised error goes below this value we stop")
parser.add_argument("--inner_sink_iter", type=int, required=True,
                    help="# of Sinkhorn iterations")
parser.add_argument("--inner_l2_loss_lambda", type=float, required=True,
                    help="Boost to use for the inner l2loss")
parser.add_argument("--crayon_send_stats_iters", type=int, default=20,
                    help="How many iters we send stats to tensorboard")
parser.add_argument("--n_models_saved", type=int, default=50,
                    help="How many snapshot of intermediate models we save")
args = parser.parse_args()
logger = get_logger("supervised_dual_space")

if args.today is None:
    args.today = datetime.date.today().strftime("%Y-%m-%d")

logger.info(f"Using experiment date: {args.today}")

# Experiment Name
exp_name = f"supervised_dual_{args.n_init}_ntrain_{args.n_train}_ntraintmap_{args.n_train_tmap}_eps_{args.epsilon}_mxinneriter_{args.max_inner_iter}_mxinnererror_{args.max_inner_error}_innsink_{args.inner_sink_iter}_innlambda_{args.inner_l2_loss_lambda}_{args.today}"
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
mets = {"initialization_loss": 0.0,
        "identity_loss": 0.0,
        "potential_l2loss": 0.0,
        "marginal_constraint_f": 0.0,
        "marginal_constraint_g": 0.0,
        "l2_loss_tm": 0.0
        }

# initialization & training iterations
N_init = args.n_init
logger.info(f"Will train initialization for {N_init} iterations")

N_train = args.n_train
logger.info(f"Will train potentials for {N_train} iterations")

max_inner_iter = args.max_inner_iter
logger.info(f"Each supervised training loop will last for at most {max_inner_iter} iterations")

max_inner_error = args.max_inner_error
logger.info(f"Each supervised training loop will stop if error gets <= {max_inner_error}")

inner_sink_iter = args.inner_sink_iter
logger.info(f"Each invocation of Sinkhorn algorithm will use {args.inner_sink_iter} iterations")

inner_l2_loss_lambda = args.inner_l2_loss_lambda
logger.info(f"Boost used when training the inner l2_loss: {inner_l2_loss_lambda}")

N_train_tMap = args.n_train_tmap
logger.info(f"Will train transport map for {N_train_tMap} iterations")

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
pF = PotentialFunction(space_dim=2, layers_dim=[128, 64])
pG = PotentialFunction(space_dim=2, layers_dim=[128, 64])
opt_potential = torch.optim.Adagrad(list(pF.parameters()) +
                                    list(pG.parameters()), lr=1e-2)

neural_map = NeuralTransportMap(space_dim=2, layers_dim=[128, 64])
opt_tm = torch.optim.Adagrad(neural_map.parameters(), lr=1e-2)

# how many times we send stats to tensorboard
n_stats_to_tensorboard = args.crayon_send_stats_iters
logger.info(f"Sending stats to tensorboard every {n_stats_to_tensorboard} iterations")

# how many times we save model
n_save: int = round(args.n_train / args.n_models_saved)
logger.info(f"Save models every {n_save} iterations, for a total of {args.n_models_saved}")


# helper function to do Sinkhorn iterations with an initial condition
def sink_with_initial_condition(u0: Tensor, v0: Tensor, C: Tensor,
                                Niter: int, sink_reg: float = 1e-2) -> \
        Tuple[Tensor, Tensor]:
    """
    Perform Sinkhorn iterations with an initial condition
    :param u0: initial condition for u
    :param v0: /// for v
    :param C: cost matrix
    :param Niter: number of iterations
    :param sink_reg: Sinkhorn regularization
    :return: updated u, v
    """

    # initialize the marginals
    a = torch.ones_like(u0)
    b = torch.ones_like(v0)

    # Gibbs matrix
    K = torch.exp(- C / sink_reg)

    # initial condition w/o backprop
    v = v0.clone().detach()
    u = u0.clone().detach()
    for _ in range(Niter):
        u: Tensor = a / torch.matmul(K, v)
        v: Tensor = b / torch.matmul(K.t(), u)
            
    return u, v


def sink_comp_constraint(f: Tensor, g: Tensor, C: Tensor,
                         sink_reg: float = 1e-2) -> Tuple[Tensor, Tensor]:
    """
    Compute constraint violation induced by the logs f, g of
    potentials
    :param f:
    :param g:
    :param C:
    :param sink_reg:
    :return:
    """

    # Gibbs matrix
    K = torch.exp(- C / sink_reg)

    # initial condition w/o backprop
    u = torch.exp(f.clone().detach() / sink_reg)
    v = torch.exp(g.clone().detach() / sink_reg)

    onU = u.view(-1) * torch.matmul(K, v).view(-1)
    onV = v.view(-1) * torch.matmul(K.t(), u).view(-1)

    return onU, onV


# initialize potentials to be 0
for iteration, data_dict in enumerate(dataloader_init):
    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)

    f = pF(X)
    g = pG(Y)
    
    torch_losses = {
        "initialization_loss" : ((f*f).mean() +
                                 (g*g).mean())
        }

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_potential,
                           loss_names=["initialization_loss"])
    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["initialization_loss"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["initialization_loss"],
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

    f = pF(X)
    g = pG(Y)

    # go to exponential space to do sinkhorn iterations
    u = torch.exp(f / epsilon)
    v = torch.exp(g / epsilon)

    # do the sinkhorn iterations
    u_next, v_next = sink_with_initial_condition(u, v, Cdist, inner_sink_iter, epsilon)

    # go to log space & normalize
    f_next = torch.log(u_next) * epsilon
    f_mean = f_next.mean()
    f_next = f_next - f_mean
    g_next = torch.log(v_next) * epsilon + f_mean

    # keep track of marginal violations
    onU, onV = sink_comp_constraint(f_next, g_next, Cdist, epsilon)
    marginal_constraint_f = (onU - 1.0).abs().mean()
    marginal_constraint_g = (onV - 1.0).abs().mean()

    # Errors used to determined when to stop
    fMax = float((f - f_next).abs().max())
    gMax = float((g - g_next).abs().max())

    # variable to decide when to stop in the inner loop
    local_counter: int = 0
    while max(fMax, gMax) > max_inner_error and local_counter < max_inner_iter:

        f = pF(X)
        g = pG(Y)
        fMax = float((f - f_next).abs().max())
        gMax = float((g - g_next).abs().max())

        # keep track of constraints
        onU, onV = sink_comp_constraint(f, g, Cdist, epsilon)
        marginal_constraint_f = torch.min(torch.tensor(1.0), (onU - 1.0).abs().mean())
        marginal_constraint_g = torch.min(torch.tensor(1.0), (onV - 1.0).abs().mean())

        torch_losses = {
            "potential_l2loss": inner_l2_loss_lambda * (l2_loss_simple_vectors_fn(f, f_next) +
                                                        l2_loss_simple_vectors_fn(g, g_next)),
            "marginal_constraint_f": marginal_constraint_f,
            "marginal_constraint_g": marginal_constraint_g,
        }

        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_potential,
                               loss_names=["potential_l2loss"],
                               retain_graph=True)

        roll_average(loss_dict=torch_losses, mets_dict=mets,
                     metrics=["potential_l2loss", "marginal_constraint_f",
                              "marginal_constraint_g"],                              
                     iteration=step_counter)

        if (step_counter + 1) % n_stats_to_tensorboard == 0:
            crayon_ship_metrics(ccexp, mets, ["potential_l2loss", "marginal_constraint_f",
                                              "marginal_constraint_g"],
                                step_counter)

        if (step_counter + 1) % n_save == 0:
            torch.save(pF.state_dict(), f"{save_dir}/pF_{step_counter}.model")
            torch.save(pG.state_dict(), f"{save_dir}/pG_{step_counter}.model")
            eval_iters.append(step_counter)

        local_counter += 1
        step_counter += 1

    if step_counter > N_train:
        break

# initialize the transport network to the identity
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

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["identity_loss"],
                            iteration)
    torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_initialized.model")

#################################################
# For each iteration we evaluate on
# we need to load the potentials & retrain the
# transport map
##################################################
# local counter to keep track of steps across
# the multiple re-trainings
iter_counter = 0
for step in eval_iters:
    # load pF & pG
    pF.load_state_dict(
        torch.load(f"{save_dir}/pF_{step}.model"))
    pG.load_state_dict(
        torch.load(f"{save_dir}/pG_{step}.model"))
    pF.eval()
    pG.eval()

    # we regenerate the loader each time
    crc_dist_tMap = CircDistros(size=N_train_tMap, sample_size=256)
    dataloader_train_tMap = DataLoader(dataset=crc_dist_tMap, batch_size=1,
                                       shuffle=True, drop_last=True)
    for iteration, data_dict in enumerate(dataloader_train_tMap):
        X = data_dict["X"].squeeze(dim=0)
        Y = data_dict["Y"].squeeze(dim=0)

        # compute cost
        Cdist = square_distances_fn(X, Y).detach()

        # compute potentials
        f = pF(X).view((-1, 1))
        g = pG(Y).view((1, -1))

        # compute probability of association
        pi = torch.exp((-Cdist + f + g) / epsilon)

        # rescale to a probability
        pi = pi / torch.sum(pi, dim=1).view((-1, 1))

        # find the average target point predicted by pi
        Yavg = torch.matmul(pi, Y).detach()

        # fit the loss
        TX = neural_map(X)

        torch_losses = {
            # using sum for l2 loss speeds up convergence
            "l2_loss_tm": ((TX-Yavg)*(TX-Yavg)).sum()
        }

        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_tm,
                               loss_names=["l2_loss_tm"]
                               )

        roll_average(loss_dict=torch_losses, mets_dict=mets,
                     metrics=["l2_loss_tm"],
                     iteration=iteration)

        if (iteration + 1) % n_stats_to_tensorboard == 0:
            crayon_ship_metrics(ccexp, mets, ["l2_loss_tm"],
                                iter_counter)
        iter_counter += 1

    mets["l2_loss_tm"] = 0.0
    torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_{step}.model")

evaluate(eval_iters, neural_map, save_dir, export_dir, plots_dir,
         crc_final)






