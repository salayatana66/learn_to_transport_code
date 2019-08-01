"""
Adversarial optimal transport

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
from prob_dist_meas import l2_loss as l2_loss_fn
from prob_dist_meas import critic_loss as critic_loss_fn

# The transporter
from estimators import NeuralTransportMap
# The adversary
from estimators import BoundedFunction

from pycrayon import CrayonClient

from typing import List

# evaluation framework
from common_evaluation import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--today", type=str, default=None)
parser.add_argument("--base_plots_dir", type=str, default="~/opt_w_dl_plots_1")
parser.add_argument("--base_models_dir", type=str, default="~/opt_w_dl_models_1")
parser.add_argument("--base_evaluation_dir", type=str, default="~/opt_w_dl_evaluation_1")
parser.add_argument("--numpy_seed", type=int, default=15)
parser.add_argument("--n_init", type=int, required=True, help="Iterations to initialize transport map")
parser.add_argument("--n_train", type=int, required=True)
parser.add_argument("--b_final_eval", type=int, default=1000, help="Batch size for final evaluation")
parser.add_argument("--lambda_critic", type=float, required=True, help="Inflating factor for the critic loss")
parser.add_argument("--n_critic", type=int, required=True, help="Number of critic iterations in the inner loop")
parser.add_argument("--grad_clip", type=float, default=None)
parser.add_argument("--crayon_send_stats_iters", type=int, default=20,
                    help="How many iters we send stats to tensorboard")
parser.add_argument("--n_models_saved", type=int, default=50,
                    help="How many snapshot of intermediate models we save")
args = parser.parse_args()
logger = get_logger("adversarial_transport")

if args.today is None:
    args.today = datetime.date.today().strftime("%Y-%m-%d")

logger.info(f"Using experiment date: {args.today}")

# Experiment Name
exp_name = f"adversarial_transport_ninit_{args.n_init}_ntrain_{args.n_train}_lambda_{args.lambda_critic}_ncritic_{args.n_critic}_gradclip_{args.grad_clip}_{args.today}"
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
mets = {"zero_loss": 0.0,
        "identity_loss": 0.0,
        "l2_loss": 0.0,
        "critic_loss": 0.0,
        "total_loss": 0.0
        }
logger.info(f"Metrics which will be logged: {mets.keys()}")

# initialization & training iterations
N_init = args.n_init
logger.info(f"Will train initialization for {N_init} iterations")

N_train = args.n_train
logger.info(f"Will train transport map for {N_train} iterations")

B_final_eval = args.b_final_eval
logger.info(f"Final batch size for evaluations & movie will be {B_final_eval}")

lambda_critic = args.lambda_critic
logger.info(f"Using lambda parameter to inflate flow loss: {args.lambda_critic}")

n_critic = args.n_critic
logger.info(f"Doing {args.n_critic} n_critic iterations in the inner loop for the adversary")

grad_clip = args.grad_clip
logger.info(f"Using {args.grad_clip} for gradient clipping")

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
neural_function = BoundedFunction(space_dim=2, layers_dim=[128, 64])
transport_map = NeuralTransportMap(space_dim=2, layers_dim=[128, 64])
opt_critic = torch.optim.Adagrad(neural_function.parameters(), lr=1e-2)
opt_tm = torch.optim.Adagrad(transport_map.parameters(), lr=1e-2)

# how many times we send stats to tensorboard
n_stats_to_tensorboard = args.crayon_send_stats_iters
logger.info(f"Sending stats to tensorboard every {n_stats_to_tensorboard} iterations")

# how many times we save model
n_save: int = round(args.n_train / args.n_models_saved)
logger.info(f"Save models every {n_save} iterations, for a total of {args.n_models_saved}")

# initialize transport to the identity
# initialize bounded function to 0
for iteration, data_dict in enumerate(dataloader_init):
    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)
    TX = transport_map(X)

    torch_losses = {
        "identity_loss": identity_loss_fn(X, TX),
        "zero_loss": neural_function(X).abs().mean() + \
                      neural_function(Y).abs().mean()
        }

    torch_losses_take_step(loss_dict=torch_losses,
                           optimizer=opt_tm,
                           loss_names=["identity_loss"])

    torch_losses_take_step(loss_dict=torch_losses, optimizer=opt_critic,
                           loss_names=["zero_loss"], retain_graph=True)

    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["identity_loss", "zero_loss"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["identity_loss"],
                            iteration)

# decide whether to apply gradient clipping
if grad_clip is not None:
    for p in neural_function.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -grad_clip, grad_clip))

# iterations to evaluate on
eval_iters: List[int] = []
# training loop
for iteration, data_dict in enumerate(dataloader_train):

    X = data_dict["X"].squeeze(dim=0)
    Y = data_dict["Y"].squeeze(dim=0)
    TX = transport_map(X)

    torch_losses = {
        "critic_loss": lambda_critic * critic_loss_fn(TX, Y, neural_function),
        "l2_loss": l2_loss_fn(TX, X)
    }

    torch_losses["total_loss"] = torch_losses["l2_loss"] + \
        torch_losses["critic_loss"]

    if iteration % n_critic == 0:
        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_tm,
                               loss_names=["total_loss"])
    else:
        torch_losses_take_step(loss_dict=torch_losses,
                               optimizer=opt_critic,
                               loss_names=["critic_loss"], minimize=False)

    roll_average(loss_dict=torch_losses, mets_dict=mets,
                 metrics=["critic_loss",
                          "l2_loss", "total_loss"],
                 iteration=iteration)

    if (iteration + 1) % n_stats_to_tensorboard == 0:
        crayon_ship_metrics(ccexp, mets, ["critic_loss", "l2_loss",
                                          "total_loss"],
                            iteration)
    if (iteration + 1) % n_save == 0:
        torch.save(transport_map.state_dict(), f"{save_dir}/neural_map_{iteration}.model")
        eval_iters.append(iteration)

# final evaluation
evaluate(eval_iters, transport_map, save_dir, export_dir, plots_dir,
         crc_final)

