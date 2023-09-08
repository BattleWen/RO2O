# Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness (RO2O)

## Getting started
The training environment (PyTorch and dependencies) can be installed as follows:

```bash
cd RO2O
conda activate -n ro2o python=3.8
pip install -r requirements/requirements_dev.txt

```

## Train
### Hyperparameters
$\eta_1$ : beta_Qsmooth == 0.0001 \
$\eta_2$ : beta_ood \
$\eta_3$ : beta_policy \
$\epsilon_{\mathrm{Q}}$ = q_smooth_eps \
$\epsilon_{\mathrm{P}}$ = policy_smooth_eps \
$\epsilon_{\text {ood}}$ = ood_smooth_eps \
$\alpha$ = q_ood_uncertainty_reg -> q_ood_uncertainty_reg_min(q_ood_uncertainty_decay) \
$n$ = sample_size

### Offline Training
MuJoco tasks:

```bash
cd algorithms/offline
python ro2o_mujoco.py --env_name walker2d-medium-v2 --beta_policy 1.0 --beta_ood 0.1 --q_smooth_eps 0.01 --policy_smooth_eps 0.01 --ood_smooth_eps 0.01 --sample_size 20 --q_ood_uncertainty_reg 1.0 --q_ood_uncertainty_reg_min 0.1 --q_ood_uncertainty_decay 5e-7 --train_seed 42
```

Antmaze tasks:
1. policy objective: LCB

```bash
cd algorithms/offline
python ro2o_antmaze_LCB.py --env_name antmaze-umaze-v2 --beta_bc 5.0 --beta_policy 1.0 --beta_ood 0.3 --q_smooth_eps 0.0 --policy_smooth_eps 0.005 --ood_smooth_eps 0.01 --sample_size 20 --q_ood_uncertainty_reg 1.0 --q_ood_uncertainty_reg_min 1.0 --q_ood_uncertainty_decay 0.0 --train_seed 42
```
2. policy objective: min

```bash
cd algorithms/offline
python ro2o_antmaze_min.py --env_name antmaze-large-play-v2 --beta_bc 1.0 --beta_policy 1.0 --beta_ood 0.5 --q_smooth_eps 0.0 --policy_smooth_eps 0.005 --ood_smooth_eps 0.01 --sample_size 20 --q_ood_uncertainty_reg 2.0 --q_ood_uncertainty_reg_min 1.0 --q_ood_uncertainty_decay 1e-6 --train_seed 42
```

### Online Training
\$uuid\$ is the pre-trained model uuid ; set q_ood_uncertainty_decay = 0, q_ood_uncertainty_reg = q_ood_uncertainty_reg_min

MuJoco tasks:
```bash
cd algorithms/finetune
python ro2o_ft_mujoco.py --env_name walker2d-medium-v2 --beta_policy 1.0 --beta_ood 0.1 --q_smooth_eps 0.01 --policy_smooth_eps 0.01 --ood_smooth_eps 0.01 --sample_size 20 --q_ood_uncertainty_reg 0.1 --q_ood_uncertainty_reg_min 0.1 --q_ood_uncertainty_decay 0 --load_path './checkpoints/RO2O-walker2d-medium-v2-$uuid$/2499.pt' --online_ft_seed 42
```
Antmaze tasks:
1. policy objective: LCB

```bash
cd algorithms/offline
python ro2o_ft_antmaze_LCB.py --env_name antmaze-umaze-v2 --online_ft_seed 42
```

2. policy objective: min
```bash
cd algorithms/offline
python ro2o_ft_antmaze_LCB.py --env_name antmaze-large-play-v2 --online_ft_seed 42
```