# Inspired by:
# 1. paper for SAC-N: https://arxiv.org/abs/2110.01548
# 2. implementation: https://github.com/snu-mllab/EDAC
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import os
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
from tqdm import trange
import wandb
import yaml
import argparse
from typing import Optional
import torch.nn.functional as F
import h5py
import time

@dataclass
class TrainConfig:
    # wandb params
    project: str = "RO2O"
    group: str = "RO2O-offline"
    name: str = "RO2O-offline"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 2_000_000
    env_name: str = "walker2d-medium-v2"
    batch_size: int = 256
    num_epochs: int = 2500
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    beta_policy: float = 1.0
    beta_ood: float = 0.1
    q_smooth_eps: float = 0.01
    policy_smooth_eps: float = 0.01 
    ood_smooth_eps: float = 0.01
    sample_size: int = 20
    q_ood_uncertainty_reg: float = 1.0
    q_ood_uncertainty_reg_min: float = 0.1
    q_ood_uncertainty_decay: float = 5e-7
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = './checkpoints/'
    data_path: str = '/mnt/data/optimal/shijiyuan/kang/.d4rl/datasets'
    deterministic_torch: bool = False
    train_seed: int = 32
    eval_seed: int = 24
    log_every: int = 100
    device: str = "cuda:0"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}-{self.train_seed}"
        self.group = f"{self.group}-{self.env_name}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

# general utils
TensorBatch = List[torch.Tensor]

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda:0",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        raise NotImplementedError


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        std = torch.exp(log_sigma)
        policy_dist = Normal(mu, std)

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)

        return tanh_action * self.max_action, log_prob, mu, std

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


class RO2O:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: VectorizedCritic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        beta_policy: float = 1.0,
        beta_ood: float = 0.1,
        q_smooth_eps: float = 0.01,
        policy_smooth_eps: float = 0.01,
        ood_smooth_eps: float = 0.01,
        sample_size: int = 20,
        q_ood_uncertainty_reg: float = 2.0,
        q_ood_uncertainty_reg_min: float = 0.1,
        q_ood_uncertainty_decay: float = 1e-6,
        alpha_learning_rate: float = 1e-4,
        device: str = "cuda:0",  
    ):
        self.device = device

        self.actor = actor
        self.critic = critic
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma

        self.beta_policy = beta_policy
        self.beta_ood = beta_ood

        self.q_smooth_eps = q_smooth_eps
        self.policy_smooth_eps = policy_smooth_eps
        self.ood_smooth_eps = ood_smooth_eps
        self.sample_size = sample_size

        self.q_ood_uncertainty_reg = q_ood_uncertainty_reg
        self.q_ood_uncertainty_reg_min = q_ood_uncertainty_reg_min
        self.q_ood_uncertainty_decay = q_ood_uncertainty_decay

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob, _ , _ = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor, action: torch.tensor) -> Tuple[torch.Tensor]:
        pi, pi_log_prob, _ , _ = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, pi) 
        
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -pi_log_prob.mean().item()

        assert pi_log_prob.shape == q_value_min.shape
        q_loss = (self.alpha * pi_log_prob - q_value_min).mean()
        policy_loss = self.get_policy_loss(state, action)

        loss = q_loss + policy_loss

        return loss, q_loss, policy_loss, batch_entropy, q_value_std

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob, _ , _ = self.actor(
                next_state, need_log_prob=True
            )
            q_next = self.target_critic(next_state, next_action)
            q_next= q_next.min(dim=0).values
            q_next = q_next - self.alpha * next_action_log_prob

            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)

        q_values = self.critic(state, action)
        
        critic_loss1 = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        noise_Q_loss = self.get_qf_loss(state, action)
        ood_loss = self.get_ood_loss(state, action)

        loss = critic_loss1 + noise_Q_loss + ood_loss

        return loss, critic_loss1, noise_Q_loss, ood_loss, q_values.mean(), q_target.mean()

    def get_noised_obs(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        M, N, A = state.shape[0], state.shape[1], action.shape[1]
        size = self.sample_size
        delta_s = 2 * eps * (torch.rand(size, N, device=self.device) - 0.5)
        # tmp_obs.shape, delta_s.shape = [M*size, N]
        tmp_obs = state.reshape(-1, 1, N).repeat(1, size, 1).reshape(-1, N)
        delta_s = delta_s.reshape(1, size, N).repeat(M, 1, 1).reshape(-1, N)
        # noised_obs.shape = [M*size, N]
        noised_obs = tmp_obs + delta_s

        return M, A, size, noised_obs, delta_s

    def get_qf_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        M, A, size, noised_obs, delta_s = self.get_noised_obs(state, action, eps = self.q_smooth_eps)
        # noised_obs.shape = [M*size, N] ; action = [M, A] -> [M*size, A]
        # noised_qs_pred = [num_qs, M*size]
        noised_qs_pred = self.critic(noised_obs, action.reshape(M, 1, A).repeat(1, size, 1).reshape(-1, A))
        noised_qs_pred = noised_qs_pred.reshape(self.critic.num_critics, M*size, 1)
        qs_pred = self.critic(state, action)
        # qs_pred = [num_qs, M] -> [num_qs, M, size] -> [num_qs, M*size, 1]
        # diff = [num_qs, M*size, 1] - [num_qs, M*size, 1]
        diff = noised_qs_pred - qs_pred.reshape(self.critic.num_critics, M, 1).repeat(1, 1, size).reshape(self.critic.num_critics, -1, 1)
        zero_tensor = torch.zeros(diff.shape, device=self.device)
        pos, neg = torch.maximum(diff, zero_tensor), torch.minimum(diff, zero_tensor)
        q_smooth_tau = 0.2
        noise_Q_loss = (1-q_smooth_tau) *  pos.square().mean(axis=0) + q_smooth_tau * neg.square().mean(axis=0)
        noise_Q_loss = noise_Q_loss.reshape(M, size)
        # noise_Q_loss_max = [M, ].mean()
        noise_Q_loss_max = noise_Q_loss[np.arange(M), torch.argmax(noise_Q_loss, axis=-1)].mean()
        noise_Q_loss_max = 0.0001 * noise_Q_loss_max

        return noise_Q_loss_max

    def get_policy_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        M, A, size, noised_obs, delta_s = self.get_noised_obs(state, action, eps = self.policy_smooth_eps)
        # noised_obs.shape = [M*size, N]
        # ood_actions.shape= [M*size, A] ; ood_actions_log.shape= [M*size, 1]
        ood_actions, ood_actions_log, noised_policy_mean, noised_policy_std = self.actor(noised_obs, need_log_prob=True)
        actions, actions_log, policy_mean, policy_std = self.actor(state, need_log_prob=True)
        noised_action_dist = Normal(noised_policy_mean, noised_policy_std)
        action_dist = Normal(policy_mean.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A), policy_std.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A))
        kl_loss = kl_divergence(action_dist, noised_action_dist).sum(axis=-1) + kl_divergence(noised_action_dist, action_dist).sum(axis=-1)
        kl_loss = kl_loss.reshape(M, size)
        max_id = torch.argmax(kl_loss, axis=1)
        policy_loss = kl_loss[np.arange(M), max_id].mean()
        policy_loss = self.beta_policy * policy_loss

        return policy_loss

    def get_ood_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        ood_loss = torch.zeros(1, device=self.device)[0]
        # noised_obs.shape = [M*size, N]
        M, A, size, noised_obs, delta_s = self.get_noised_obs(state, action, eps = self.ood_smooth_eps)
        # ood_actions.shape= [M*size, A]
        ood_actions, ood_action_log, _ , _ = self.actor(noised_obs, need_log_prob=True)
        # ood_qs_pred = [num_qs, M*size] -> [num_qs, M*size, 1]
        ood_qs_pred = self.critic(noised_obs, ood_actions).reshape(self.critic.num_critics, M*size, 1)
        ood_target = ood_qs_pred - self.q_ood_uncertainty_reg * ood_qs_pred.std(axis=0)
        ood_loss = F.mse_loss(ood_target.detach(), ood_qs_pred).mean()
        ood_loss = self.beta_ood * ood_loss
        self.q_ood_uncertainty_reg = max(self.q_ood_uncertainty_reg - self.q_ood_uncertainty_decay, self.q_ood_uncertainty_reg_min)

        return ood_loss

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)
        
        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, q_loss, policy_loss, batch_entropy, q_values_std = self._actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss, critic_loss1, noise_Q_loss, ood_loss, q_values, q_target = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = self.critic(state, random_actions).std(0).mean().item()

        update_info = {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "critic_loss": critic_loss.item(),
            "critic_loss1": critic_loss1.item(),
            "noise_Q_loss": noise_Q_loss.item(),
            "ood_loss": ood_loss.item(),
            "q_ood_uncertainty_reg": self.q_ood_uncertainty_reg,
            "q_values": q_values.item(),
            "q_target": q_target.item(),
            "actor_loss": actor_loss.item(),
            "q_loss": q_loss.item(),  
            # "bc_loss": bc_loss.item(),
            "policy_loss": policy_loss.item(),
            "batch_entropy": batch_entropy,
            "q_values_std": q_values_std,
            "q_random_std": q_random_std,
        }

        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.array(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def get_data(data_path):
    with h5py.File(data_path, 'r') as dataset:
        N = dataset['rewards'].shape[0]
        state_ = []
        action_ =[]
        reward_ = []
        next_state_ = []
        done_ = []

        state_ = np.array(dataset['observations'])
        action_ = np.array(dataset['actions'])
        reward_ = np.array(np.squeeze(dataset['rewards']))
        next_state_ = np.array(dataset['next_observations'])
        done_ = np.array(dataset['terminals'])

    return {
        'observations': state_,
        'actions': action_,
        'next_observations': next_state_,
        'rewards': reward_,
        'terminals': done_,
    }

@pyrallis.wrap()
def train(config: TrainConfig): 
    wandb_init(asdict(config))
    # data, evaluation, env setup
    env = gym.make(config.env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # dataset = d4rl.qlearning_dataset(env)
    split_env = config.env_name.split('-')
    if len(split_env) == 3:
        data_path = f"{config.data_path}/{split_env[0]}_{split_env[1]}_v2.hdf5"
        dataset = get_data(data_path)
    if len(split_env) == 4 and split_env[2] == "expert":
        data_path = f"{config.data_path}/{split_env[0]}_{split_env[1]}_{split_env[2]}_v2.hdf5"
        dataset = get_data(data_path)
    if len(split_env) == 4 and split_env[2] == "replay":
        data_path = f"{config.data_path}/{split_env[0]}_{split_env[1]}_{split_env[2]}_v2.hdf5"
        dataset = get_data(data_path)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean = state_mean, state_std = state_std)

    set_seed(config.train_seed, env, deterministic_torch=config.deterministic_torch)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )

    buffer.load_d4rl_dataset(dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    
    trainer = RO2O(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        beta_policy=config.beta_policy,
        beta_ood=config.beta_ood,
        q_smooth_eps=config.q_smooth_eps,
        policy_smooth_eps=config.policy_smooth_eps,
        ood_smooth_eps=config.ood_smooth_eps,
        sample_size=config.sample_size,
        q_ood_uncertainty_reg = config.q_ood_uncertainty_reg,
        q_ood_uncertainty_reg_min = config.q_ood_uncertainty_reg_min,
        q_ood_uncertainty_decay = config.q_ood_uncertainty_decay,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info}, step = total_updates)

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(env, "get_normalized_score"):
                normalized_score = env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None and epoch == config.num_epochs - 1:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
