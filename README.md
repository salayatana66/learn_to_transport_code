# learn_to_transport_code
WIP

## Structure
Code is organized in common modules & 
individual scripts for experiments. Between
individual experiment scripts code is duplicated *ad libitum*.

### Prerequisites
* `conda.yaml`: the environment *manifest* that was
used for the Python code.
* `install_pycrayon.sh`: installation for the Docker image running
TensorBoard and communicating with PyTorch via `pycrayon`.

### Common modules
* `circ_distros.py`: The source and target distributions.
* `circ_samplers.py`: Sampling utilities for the source and 
target distributions.
* `common_evaluation.py`: Utilities to evaluate the transport maps produced in
the experiments.
* `general_utils.py`: Generic utilities, e.g. creating model directories,
training PyTorch models.
* `prob_dist_meas.py`: Functions to create *distances* between
probability distributions.
* `sinkhorn.py`: A plain numpy implementation of Sinkhorn's algorithm.

