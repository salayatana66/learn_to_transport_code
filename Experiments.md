# Experiments

Code used to run the experiments.

## Algorithms based on Heuristics

### Flows
* Covariance flow run
```bash
python heuristic_covariance_flow.py --n_init 500 \
 --n_train 5000 \
 --n_models_saved 50
```

* Cost-regularized covariance flow
```bash
python heuristic_covariance_w_tpreg_flow.py --n_init 500 \
 --n_train 5000 \
 --n_models_saved 50
```

* Discrepancy @1
```bash
python heuristic_discr_flow.py --n_init 500 \
 --n_train 100000 --lambda_par 10.0 --cutoff_par 1 \
 --n_models_saved 50
```

* Discrepancy @5
```bash
python heuristic_discr_flow.py --n_init 500 \
 --n_train 40000 --lambda_par 10.0 --cutoff_par 5 \
 --n_models_saved 50
```

* Cost-regularized discrepancy @1
```bash
python heuristic_discr_w_tpreg_flow.py --n_init 500 \
 --n_train 50000 --lambda_par 10.0 --cutoff_par 1 \
 --n_models_saved 50
```

* Cost-regularized discrepancy @5 
```bash
python heuristic_discr_w_tpreg_flow.py --n_init 500 \
 --n_train 30000 --lambda_par 10.0 --cutoff_par 5 \
 --n_models_saved 50
```

* Exponential flow
```bash
python heuristic_exp_flow.py --n_init 500 \
 --n_train 30000 --lambda_par 100 \
 --n_models_saved 50
```

* Cost-regularized exponential flow
```bash
python heuristic_exp_w_tpreg_flow.py --n_init 500 \
 --n_train 30000 --lambda_par 100 \
 --n_models_saved 50
```

* Symmetric discrete @5 flow
```bash
python heuristic_symdiscr_flow.py --n_init 500 \
 --n_train 50000 --lambda_par 10 \
 --cutoff_par 5 \
 --n_models_saved 50
```

* Symmetric discrete @1 flow
```bash
python heuristic_symdiscr_flow.py --n_init 500 \
 --n_train 50000 --lambda_par 10 \
 --cutoff_par 1 \
 --n_models_saved 50
```

* Cost-regularized symmetric discrete @1 flow 
```bash
python heuristic_symdiscr_w_tpreg_flow.py --n_init 500 \
 --n_train 50000 --lambda_par 10 \
 --cutoff_par 1 \
 --n_models_saved 50
```

* Cost-regularized symmetric discrete @5 flow
```bash
python heuristic_symdiscr_w_tpreg_flow.py --n_init 500 \
 --n_train 50000 --lambda_par 10 \
 --cutoff_par 5 \
 --n_models_saved 50
```

### Adversarial Training

* Adversarial training with critic @2 and different boosts for the critic
```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic 1.0 --n_critic 2 \
 --n_models_saved 50
```

```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic 100.0 --n_critic 2 \
 --n_models_saved 50
```

```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic 10.0 --n_critic 2 \
 --n_models_saved 50
```

* Increasing iterations of critic to 10
```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic 1.0 --n_critic 10 \
 --n_models_saved 50
```

* Lowering the critic contribution
```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic .1 --n_critic 2 \
 --n_models_saved 50
```

* Using gradient clipping
```bash
python adversarial_transport.py --n_init 500 \
 --n_train 50000 --lambda_critic 1.0 --n_critic 2 \
 --n_models_saved 50 \
 --grad_clip 1e-2
```

## Methods with a "rigorous" mathematical justification

### Dual training after Seguy et. alii

* Entropic regularization
```bash
python dual_transport_seguy.py --n_init 500 \
  --n_train 10000 --n_train_tmap 5000 --epsilon 0.1 \
  --regularization ent \
  --reg_sum_or_mean mean \
  --n_models_saved 50
```

```bash
python dual_transport_seguy.py --n_init 500 \
 --n_train 10000 --n_train_tmap 5000 --epsilon 0.1 \
 --regularization ent \
 --reg_sum_or_mean sum \
 --n_models_saved 50
```

* l2-regularization

```bash
python dual_transport_seguy.py --n_init 500 \
 --n_train 5000 --n_train_tmap 5000 --epsilon 0.1 \
 --regularization l2 \
 --reg_sum_or_mean mean \
 --n_models_saved 50
```

```bash
python dual_transport_seguy.py --n_init 500 \
 --n_train 5000 --n_train_tmap 5000 --epsilon 0.1 \
 --regularization l2 \
 --reg_sum_or_mean sum \
 --n_models_saved 50
```

### Supervised learning

* Training directly the transport map
```bash
python supervised_map.py --n_init 500 \
--n_train 50000 --epsilon 0.05 \
--max_inner_iter 1000 \
--inner_sink_iter 1000 \
--n_models_saved 50  
```

```bash
python supervised_map.py --n_init 500 \
--n_train 50000 --epsilon 0.05 \
--max_inner_iter 200 \
--inner_sink_iter 1000 \
--n_models_saved 50  
```

* Training the potentials
```bash
python supervised_dual_space.py --n_init 500 \
--n_train 20000 --n_train_tmap 10000 --epsilon 0.1 \
--max_inner_iter 1000 \
--max_inner_error 0.01 \
--inner_sink_iter 1000 \
--inner_l2_loss_lambda 100.0 \
--n_models_saved 50
```

```bash
python supervised_dual_space.py --n_init 500 \
--n_train 30000 --n_train_tmap 5000 --epsilon 0.05 \
--max_inner_iter 1000 \
--max_inner_error 0.01 \
--inner_sink_iter 1000 \
--inner_l2_loss_lambda 100.0 \
--n_models_saved 50
```

* Training the transport plan

```bash
  python supervised_prob_space.py --n_init 500 \
--n_train 50000 --n_train_tmap 5000 --epsilon 0.1 \
--max_inner_iter 1000 \
--inner_sink_iter 1000 \
--n_models_saved 50 
```
