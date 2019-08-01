"""
Common evaluation of transport maps produced by the algorithms
@author: Andrea Schioppa
"""
import sys
import os
sys.path.insert(0, os.path.expanduser("~/learn_to_transport_code"))

import os
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from general_utils import visualize

# the sinkhorn algorithm
from sinkhorn import compute_pairwise_euclidean
from sinkhorn import sinkhorn_iterations

from prob_dist_meas import l2_loss as l2_loss_fn
from prob_dist_meas import mean_discrepancy as mean_discrepancy_fn

from typing import List, Dict


def evaluate(eval_iters: List[iter], neural_map: torch.nn.Module,
             model_dir: str, export_dir: str, plot_dir: str,
             dataset_final_evaluation: Dataset) -> None:
    """
    Evaluate quality of transport maps produced at different iteration
    points
    :param eval_iters: the iteration points we evaluate at
    :param neural_map: the (neural) transport map
    :param model_dir: directory containing model states to restore
    :param export_dir: directory to which we will export the .csv evaluation metrics
    :param plot_dir: directory we will use to generate movies
    :param dataset_final_evaluation: dataset on which we perform the evaluation
    """

    # evaluation metrics
    eval_metrics = {"root_mean_discrepancy@1": [],
                    "l2cost": [],
                    "optimal_wasserstein_distance": [],
                    "root_cost_X_TX": [],
                    "wasserstein_distance_TX_Y": [],
                    "iteration": [],
                    "l2dist_to_optimal_map": [],
    }

    for iteration in eval_iters:

        # extract data to evaluate at
        data_dict = dataset_final_evaluation[0]

        X = data_dict["X"].squeeze(dim=0)
        Y = data_dict["Y"].squeeze(dim=0)

        # reload model state
        neural_map.load_state_dict(
            torch.load(f"{model_dir}/neural_map_{iteration}.model"))
        neural_map.eval()
        
        TX = neural_map(X)

        # discrepancy as distance of the supports of the measures
        # Y & TX
        mean_discrepancy = mean_discrepancy_fn(Y, TX, 1)

        # cost X -> TX via the map TX
        l2cost = float(l2_loss_fn(X, TX))

        # compute sinkhorn distance Y -- TX
        # independent assessment using a python loop and
        # a low value of regularization
        Xnp = X.detach().numpy()
        Ynp = Y.detach().numpy()
        TXnp = TX.detach().numpy()
        C = compute_pairwise_euclidean(Ynp, TXnp)
        a = np.ones((Ynp.shape[0],)) / Ynp.shape[0]
        b = np.ones((Ynp.shape[0],)) / Ynp.shape[0]
        P, _, _ = sinkhorn_iterations(100, a, b, C, 1e-2)

        sink_ytx = np.mean(np.sum(P * C, axis=1) / a)

        # compute sinkhorn X -- Y
        C = compute_pairwise_euclidean(Xnp, Ynp)
        a = np.ones((Ynp.shape[0],)) / Ynp.shape[0]
        b = np.ones((Ynp.shape[0],)) / Ynp.shape[0]
        P, _, _ = sinkhorn_iterations(100, a, b, C, 1e-2)
        sink_yx = np.mean(np.sum(P * C, axis=1) / a)

        # compute the map optimal map
        # via the weighted baricenters on fibers X \times Y -> Y
        Tpx = P.dot(Ynp[:, 0]) / a
        Tpy = P.dot(Ynp[:, 1]) / b

        Tp = np.stack((Tpx, Tpy), axis=1)
        l2_dist_to_optimal_map = np.sqrt(np.power(Tp - TXnp, 2.0).sum(axis=1).mean(axis=0))

        # ship metrics
        eval_metrics["iteration"].append(iteration)
        eval_metrics["root_mean_discrepancy@1"].append(float(np.sqrt(float(mean_discrepancy))))
        eval_metrics["l2cost"].append(l2cost)
        eval_metrics["root_cost_X_TX"].append(float(np.sqrt(l2cost)))
        eval_metrics["wasserstein_distance_TX_Y"].append(float(np.sqrt(sink_ytx)))
        eval_metrics["optimal_wasserstein_distance"].append(float(np.sqrt(sink_yx)))
        eval_metrics["l2dist_to_optimal_map"].append(float(l2_dist_to_optimal_map))
        print(eval_metrics)

        # create a visualization
        visualize(X, Y, TX, plot_dir + "/" + "vis_new" 
                  + f"{iteration+1:06d}" + ".png")

    print("Creating final movie")
    os.chdir(plot_dir)

    # this command combines .pngs into a movie
    subprocess.check_output(["convert", "-set", "delay", "100",
                             "-colorspace", "RGB", "-colors", "256", "-dispose",
                             "1", "-loop", "0", "-scale", "100%", "*.png", "Output.gif"])

    # save metrics
    outPandas = pd.DataFrame(eval_metrics)
    outPandas.to_csv(f"{export_dir}/evaluation_metrics.csv", header=True, index=False)

