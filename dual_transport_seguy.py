"""
Learning a transport map
solving the regularized dual problem
following Seguy et alii

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
from prob_dist_meas import identity_loss as identity_loss_fn
from prob_dist_meas import square_distances as square_distances_fn

# The transporter
from estimators import NeuralTransportMap
# The potential function
from estimators import PotentialFunction

from pycrayon import CrayonClient

from typing import List, Optional

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
parser.add_argument("--regularization", type=str, required=True,
                    choices=["ent", "l2"])
parser.add_argument("--reg_sum_or_mean", type=str, required=True,
                    choices=["sum", "mean"],
                    help="Apply regularization as mean (the math is right) or Seguy (sum, maybe their math is wrong?)")
parser.add_argument("--crayon_send_stats_iters", type=int, default=20,
                    help="How many iters we send stats to tensorboard")
parser.add_argument("--n_models_saved", type=int, default=50,
                    help="How many snapshot of intermediate models we save")
args = parser.parse_args()
logger = get_logger("dual_transport_seguy")

if args.today is None:
    args.today = datetime.date.today().strftime("%Y-%m-%d")

logger.info(f"Using experiment date: {args.today}")

# Experiment Name
exp_name = f"dual_seguy_{args.n_init}_ntrain_{args.n_train}_ntraintmap_{args.n_train_tmap}_eps_{args.epsilon}_regtype_{args.regularization}_regagg_{args.reg_sum_or_mean}_{args.today}"
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
mets = {"uX": 0.0,
        "vY": 0.0,
        "total_potential": 0.0,
        "potential_u_initialization_loss": 0.0,
        "potential_v_initialization_loss": 0.0,
        "weighted_barycentric": 0.0,
        "identity_loss": 0.0
        }

reg_name: Optional[str] = None
if args.regularization == "ent":
    logger.info("Using entropy (so exponential) regularization")
    reg_name = "expPen"
elif args.regularization == "l2":
    logger.info("Using l2 regularization")
    reg_name = "l2Pen"

mets[reg_name] = 0.0

logger.info(f"Using {args.reg_sum_or_mean} to aggregate the regularization")

# initialization & training iterations
N_init = args.n_init
logger.info(f"Will train initialization for {N_init} iterations")

N_train = args.n_train
logger.info(f"Will train potentials for {N_train} iterations")

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
potential_u = PotentialFunction(space_dim=2, layers_dim=[128, 64])
potential_v = PotentialFunction(space_dim=2, layers_dim=[128, 64])
opt_potential = torch.optim.Adagrad(list(potential_u.parameters()) +
                                    list(potential_v.parameters()),
                                    lr=1e-2)
neural_map = NeuralTransportMap(space_dim=2, layers_dim=[128, 64])
opt_tm = torch.optim.Adagrad(neural_map.parameters(), lr=1e-2)

# how many times we send stats to tensorboard
n_stats_to_tensorboard = args.crayon_send_stats_iters
logger.info(f"Sending stats to tensorboard every {n_stats_to_tensorboard} iterations")

# how many times we save model
n_save: int = round(args.n_train / args.n_models_saved)
logger.info(f"Save models every {n_save} iterations, for a total of {args.n_models_saved}")

# initialize potentials to 0
for iteration, data_dict in enumerate(dataloader_init):
    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)

    torch_losses = {
        "potential_u_initialization_loss": identity_loss_fn(0.0, potential_u(X)),
        "potential_v_initialization_loss": identity_loss_fn(0.0, potential_v(Y))
    }

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_potential,
                           loss_names=["potential_u_initialization_loss",
                                       "potential_v_initialization_loss"])
    
    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["potential_u_initialization_loss",
                          "potential_v_initialization_loss"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["identity_loss"],
                            iteration)


# iterations to evaluate on
eval_iters: List[int] = []

# layer to use with l2
reluLayer = torch.nn.ReLU()
# training loop for the dual
for iteration, data_dict in enumerate(dataloader_train):

    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)

    # find the constraint violation
    C = square_distances_fn(X, Y)
    uX = potential_u(X).view((-1, 1))
    vY = potential_v(Y).view((1, -1))
    cons_violation = uX + vY - C

    # Regularization potential
    regTerm: Optional[torch.Tensor] = None
    if args.regularization == "ent":
        regTerm = -epsilon * torch.exp(cons_violation / epsilon)
    elif args.regularization == "l2":
        cons_violation = reluLayer(cons_violation)
        regTerm = - (cons_violation * cons_violation) / (4 * epsilon)

    # Aggregation
    regAgg: Optional[torch.Tensor] = None
    if args.reg_sum_or_mean == "sum":
        regAgg = regTerm.sum()
    elif args.reg_sum_or_mean == "mean":
        regAgg = torch.mean(regTerm, dim=1).sum()

    torch_losses = {"uX": uX.sum(), "vY": vY.sum()}

    torch_losses[reg_name] = regAgg

    # signs get reversed because of minimization (paper maximizes)
    torch_losses["total_potential"] = - (torch_losses["uX"] + torch_losses["vY"] + \
                                         torch_losses[reg_name])

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_potential,
                           loss_names=["total_potential"])

    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=[reg_name, "uX", "vY", "total_potential"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, [reg_name, "uX", "vY", "total_potential"],
                            iteration)

    if (iteration + 1) % n_save == 0:
        torch.save(potential_u.state_dict(), f"{save_dir}/pU_{iteration}.model")
        torch.save(potential_v.state_dict(), f"{save_dir}/pV_{iteration}.model")
        eval_iters.append(iteration)


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
    torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_initialized.model")

#################################################
# For each iteration we evaluate on
# we need to load the potentials & retrain the
# transport map
##################################################
# local counter to keep track of steps across
# the multiple retrainings
iter_counter = 0
for step in eval_iters:
    print(f"step in eval_iters: {step}")

    # load saved model
    potential_u.load_state_dict(
        torch.load(f"{save_dir}/pU_{step}.model"))
    potential_u.eval()

    potential_v.load_state_dict(
        torch.load(f"{save_dir}/pV_{step}.model"))
    potential_v.eval()

    # create new dataset for training the transport map
    crc_dist_tMap = CircDistros(size=N_train_tMap, sample_size=256)
    dataloader_train_tMap = DataLoader(dataset=crc_dist_tMap, batch_size=1,
                                       shuffle=True, drop_last=True)

    for iteration, data_dict in enumerate(dataloader_train_tMap):

        X = data_dict["X"].squeeze(dim=0)
        Y = data_dict["Y"].squeeze(dim=0)
        TX = neural_map(X)
    
        # find the constraint violation
        C = square_distances_fn(X, Y)
        uX = potential_u(X).view((-1, 1))
        vY = potential_v(Y).view((1, -1))
        cons_violation = uX + vY - C
        CTmap = square_distances_fn(TX, Y)

        # weights in distance computation
        H: Optional[torch.Tensor] = None

        if args.regularization == "ent":
            H = torch.exp(cons_violation/epsilon)
        elif args.regularization == "l2":
            cons_violation = reluLayer(cons_violation)
            H = cons_violation / (2 * epsilon)

        torch_losses = {
            "weighted_barycentric": torch.mean((CTmap * H), dim=0).sum(),

        }

        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_tm,
                               loss_names=["weighted_barycentric"])

        roll_average(loss_dict=torch_losses, mets_dict=mets,
                     metrics=["weighted_barycentric"],
                     iteration=iteration)

        if (iter_counter + 1) % n_stats_to_tensorboard == 0:
            crayon_ship_metrics(ccexp, mets, ["weighted_barycentric"],
                                iter_counter)
        iter_counter += 1

    torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_{step}.model")

evaluate(eval_iters, neural_map, save_dir, export_dir, plots_dir,
         crc_final)


