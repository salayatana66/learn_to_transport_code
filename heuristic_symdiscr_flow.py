"""
Learning a transport map heuristically using
a symmetric discrepancy (~ to Hausdorff distance) flow

@author: Andrea Schioppa
"""

import numpy as np
import argparse
import datetime
import torch
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
from prob_dist_meas import mean_discrepancy as mean_discrepancy_fn

from common_evaluation import evaluate

# The transporter
from estimators import NeuralTransportMap

from pycrayon import CrayonClient

from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str, default=None)
parser.add_argument("--base_plots_dir", type=str, default="~/opt_w_dl_plots_1")
parser.add_argument("--base_models_dir", type=str, default="~/opt_w_dl_models_1")
parser.add_argument("--base_evaluation_dir", type=str, default="~/opt_w_dl_evaluation_1")
parser.add_argument("--numpy_seed", type=int, default=15)
parser.add_argument("--n_init", type=int, required=True, help="Iterations to initialize transport map")
parser.add_argument("--n_train", type=int, required=True)
parser.add_argument("--lambda_par", type=float, required=True, help="Inflating factor for the exponential flow")
parser.add_argument("--cutoff_par", type=int, required=True, help="Number of 'closest' points used in computing the discrepancy")
parser.add_argument("--b_final_eval", type=int, default=1000, help="Batch size for final evaluation")
parser.add_argument("--crayon_send_stats_iters", type=int, default=20,
                    help="How many iters we send stats to tensorboard")
parser.add_argument("--n_models_saved", type=int, default=50,
                    help="How many snapshot of intermediate models we save")
args = parser.parse_args()
logger = get_logger("heuristic_symdiscr_flow")

if args.today is None:
    args.today = datetime.date.today().strftime("%Y-%m-%d")

logger.info(f"Using experiment date: {args.today}")

# Experiment Name
exp_name = f"symdiscr_ninit_{args.n_init}_ntrain_{args.n_train}_lambda_{args.lambda_par}_cutoff_{args.cutoff_par}_{args.today}"
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
mets = {"sym_mean_discrepancy": 0.0,
        "identity_loss": 0.0
        }
logger.info(f"Metrics which will be logged: {mets.keys()}")

# initialization & training iterations
N_init = args.n_init
logger.info(f"Will train initialization for {N_init} iterations")

N_train = args.n_train
logger.info(f"Will train transport map for {N_train} iterations")

B_final_eval = args.b_final_eval
logger.info(f"Final batch size for evaluations & movie will be {B_final_eval}")

lambda_par = args.lambda_par
logger.info(f"Using lambda parameter to inflate flow loss: {args.lambda_par}")

cutoff_par = args.cutoff_par
logger.info(f"Number of closest points considered in discrepancy: {args.cutoff_par}")

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

# initialize network to the identity
for iteration, data_dict in enumerate(dataloader_init):
    X = data_dict["X"].squeeze(dim=0)
    TX = neural_map(X)

    torch_losses = {
        "identity_loss" : identity_loss_fn(X, TX)
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

# iterations to evaluate on
eval_iters: List[int] = []

# training loop
for iteration, data_dict in enumerate(dataloader_train):

    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)
    TX = neural_map(X)

    torch_losses = {
        "sym_mean_discrepancy": lambda_par * (mean_discrepancy_fn(TX, Y, cutoff=cutoff_par) +
                                              mean_discrepancy_fn(Y, TX, cutoff=cutoff_par))
    }

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_tm,
                           loss_names=["sym_mean_discrepancy"]
    )

    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["sym_mean_discrepancy"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["sym_mean_discrepancy"],
                            iteration)
    if (iteration + 1) % n_save == 0:
        torch.save(neural_map.state_dict(), f"{save_dir}/neural_map_{iteration}.model")
        eval_iters.append(iteration)

# final evaluation
evaluate(eval_iters, neural_map, save_dir, export_dir, plots_dir,
         crc_final)

