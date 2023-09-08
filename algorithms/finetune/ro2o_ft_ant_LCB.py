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
import torch.nn.functional as F
import torch.nn.utils as utils
import argparse

@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "RO2O-FT"
    name: str = "RO2O-FT"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    beta_online: float = -4.0
    beta_bc: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    buffer_size: int = 2_000_000
    env_name: str = "antmaze-umaze-v2"
    batch_size: int = 256
    normalize_reward: bool = True
    # evaluation params
    eval_episodes: int = 100
    eval_every: int = 50
    # online_ft params
    train_starts: int = 2500
    max_steps: int = 250_000
    memory_size: int = 250_000

    # general params
    checkpoints_path: Optional[str] = './checkpoints/'
    load_path: Optional[str] = './checkpoints/RO2O-antmaze-umaze-v2-49c9abcf/999.pt'
    deterministic_torch: bool = False
    eval_seed: int = 32
    online_ft_seed: int = 24
    log_every: int = 100
    device: str = "cuda:0"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
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
        # log_sigma = torch.clip(log_sigma, -20, 2)
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
        # self.critic = nn.Sequential(
        #     VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     VectorizedLinear(hidden_dim, hidden_dim, num_critics),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     VectorizedLinear(hidden_dim, hidden_dim, num_critics),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ReLU(),
        #     VectorizedLinear(hidden_dim, 1, num_critics),
        #     # nn.LayerNorm(1),
        # )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            if isinstance(layer, VectorizedLinear):
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
        beta_online: float = -4.0,
        beta_bc: float = 1.0,
        alpha_learning_rate: float = 1e-4,
        device: str = "cuda:0",  # noqa
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
        self.beta_online = beta_online
        self.beta_bc = beta_bc

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
        q_value_dist = self.critic(state, pi) - self.alpha * pi_log_prob
        assert q_value_dist.shape[0] == self.critic.num_critics
        batch_entropy = -pi_log_prob.mean().item()
        q_value_mean = q_value_dist.mean(0)
        q_value_std = q_value_dist.std(0)

        q_loss = -(q_value_mean + self.beta_online * q_value_std).mean()
        
        bc_loss = F.mse_loss(pi, action)
        # policy_loss = self.get_policy_loss(state, action)

        loss = q_loss + self.beta_bc * bc_loss #+ policy_loss

        return loss, q_loss, bc_loss, batch_entropy, q_value_std.mean().item()

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
            q_next = q_next - self.alpha * next_action_log_prob
                       
            q_target = reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * q_next.unsqueeze(-1)

            first_action, first_action_log_prob, _ , _ = self.actor(
                state, need_log_prob=True
            )
        q_values = self.critic(state, action)
        # q_values_2 = self.critic(state, first_action)

        critic_loss1 = ((q_values - q_target.squeeze(-1)) ** 2).mean(dim=1).sum(dim=0)
        # critic_loss2 = (q_values_2 - q_values).mean(dim=1).sum(dim=0)
        # noise_Q_loss = self.get_qf_loss(state, action)
        # ood_loss = self.get_ood_loss(state, action)

        loss = critic_loss1 #+ noise_Q_loss #+ ood_loss

        return loss, critic_loss1, q_values.mean(), q_target.mean()
    

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


    def update_online(self, batch: TensorBatch) -> Dict[str, float]:
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
        actor_loss, q_loss, bc_loss, batch_entropy, q_value_std = self._actor_loss(state, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss, critic_loss1, q_values, q_target = self._critic_loss(state, action, reward, next_state, done)
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

        update_online_info = {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "critic_loss": critic_loss.item(),
            "critic_loss1": critic_loss1.item(),
            # "noise_Q_loss": noise_Q_loss.item(),
            # "ood_loss": ood_loss.item(),
            # "q_ood_uncertainty_reg": q_ood_uncertainty_reg,
            "q_values": q_values.item(),
            "q_target": q_target.item(),
            "actor_loss": actor_loss.item(),
            "q_loss": q_loss.item(),  
            "bc_loss": bc_loss.item(),
            # "policy_loss": policy_loss.item(),
            "batch_entropy": batch_entropy,
            "q_value_std": q_value_std,
            # "q_policy_std": q_policy_std,
            # "q_random_std": q_random_std,
        }
        return update_online_info

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
        dataset["rewards"] = 4 * (dataset["rewards"] - 0.5)

def sample(minibatch: list) -> TensorBatch:
    state = torch.from_numpy(np.asarray(tuple(d[0] for d in minibatch), dtype=np.float32))
    action = torch.from_numpy(np.asarray(tuple(d[1] for d in minibatch), dtype=np.float32))
    reward = torch.from_numpy(np.asarray(tuple(d[2] for d in minibatch), dtype=np.float32))
    next_state = torch.from_numpy(np.asarray(tuple(d[3] for d in minibatch), dtype=np.float32))
    done = torch.from_numpy(np.asarray(tuple(d[4] for d in minibatch), dtype=np.float32))
    return [state, action, reward, next_state, done]

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

# @pyrallis.wrap()
def online_ft(config:TrainConfig):
    wandb_init(asdict(config))
    # data, evaluation, env setup
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env_name)

    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    eval_env = wrap_env(env, state_mean = state_mean, state_std = state_std)
    set_seed(config.online_ft_seed, env, deterministic_torch=config.deterministic_torch)

    states = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_states = dataset["next_observations"]
    dones = dataset["terminals"]

    replay_buffer = []

    # if you use offline_buffer
    # for j in range(len(states)):
    #     replay_buffer.append((states[j], actions[j], rewards[j], next_states[j], dones[j]))
    # replay_buffer = replay_buffer[-2500:]

    print("...data conversion complete")

    # Actor & Critic setup
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.critic_learning_rate
    )
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)

    trainer = RO2O(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        beta_online=config.beta_online,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )

    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)


    state_dict = torch.load(config.load_path, map_location = config.device)
    trainer.load_state_dict(state_dict)

    env_steps = 0
    ev_every = config.eval_every
    total_updates = 0

    while env_steps < config.max_steps + 1:
        done = False
        state = env.reset()
        state = (state - state_mean) / state_std
        step_env = 0  
        # score_train = 0
        while not done:
            # action = trainer.get_max_action(torch.tensor(state, device = config.device, dtype=torch.float32))
            action = trainer.actor.act(state, config.device)
            next_state, reward, done, info = env.step(action)
            next_state = (next_state - state_mean) / state_std
            # score_train += reward
            step_env += 1
            env_steps += 1

            if step_env == env._max_episode_steps:
                done_rb = False
                # print("Max env steps reached")
            else:
                done_rb = done
            
            replay_buffer.append((state, action, 4 * (reward - 0.5), next_state, done_rb))
            state = next_state

            if len(replay_buffer) > config.memory_size:
                replay_buffer.pop(0)
            
            # training
            if len(replay_buffer) >= config.train_starts:
                minibatch = random.sample(replay_buffer, config.batch_size)
                update_online_info = trainer.update_online(sample(minibatch))

                if total_updates % config.log_every == 0:
                    wandb.log({"epoch": env_steps // 1000, **update_online_info}, step = total_updates)

                total_updates += 1

            # evaluation
            if env_steps % ev_every == 0 and len(replay_buffer) >= config.train_starts:
                eval_returns = eval_actor(
                    env=eval_env,
                    actor=actor,
                    n_episodes=config.eval_episodes,
                    seed=config.eval_seed + config.online_ft_seed,
                    device=config.device,
                )
                eval_log = {
                    "eval/reward_mean": np.mean(eval_returns),
                    "eval/reward_std": np.std(eval_returns),
                    "epoch": env_steps // 1000,
                    "steps": total_updates
                }
                if hasattr(eval_env, "get_normalized_score"):
                    normalized_score = env.get_normalized_score(eval_returns) * 100.0
                    eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                    eval_log["eval/normalized_score_std"] = np.std(normalized_score)

                wandb.log(eval_log)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # wandb params
    parser.add_argument('--project', default="CORL", type=str)
    parser.add_argument('--group', default="RO2O-FT", type=str)
    parser.add_argument('--name', default="RO2O-FT", type=str)

    # model params
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_critics', default=10, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=5e-3, type=float)    
    parser.add_argument('--actor_learning_rate', default=3e-4, type=float)
    parser.add_argument('--critic_learning_rate', default=3e-4, type=float)
    parser.add_argument('--alpha_learning_rate', default=3e-4, type=float)
    parser.add_argument('--max_action', default=1.0, type=float)
    
    # training params
    parser.add_argument('--buffer_size', default=2_000_000, type=int)
    parser.add_argument('--env_name', default="antmaze-umaze-v2", type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--normalize_reward', default=True, type=bool)
    parser.add_argument('--train_starts', default=2500, type=int)
    parser.add_argument('--max_steps', default=250_000, type=int)
    parser.add_argument('--memory_size', default=250_000, type=int)   
    parser.add_argument('--beta_bc', default=1.0, type=float)
    parser.add_argument('--beta_online', default=-4.0, type=float)

    # evaluation params
    parser.add_argument('--eval_episodes', default=100, type=int)
    parser.add_argument('--eval_every', default=50, type=int)

    # general params
    parser.add_argument('--checkpoints_path', default='./checkpoints/', type=str)
    parser.add_argument('--load_path', default='./checkpoints/', type=str)
    parser.add_argument('--deterministic_torch', default=False, type=bool)
    parser.add_argument('--eval_seed', default=24, type=int)
    parser.add_argument('--online_ft_seed', default=24, type=int)
    parser.add_argument('--log_every', default=100, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)

    args = parser.parse_args()
    config = TrainConfig(**vars(args))

    online_ft(config)