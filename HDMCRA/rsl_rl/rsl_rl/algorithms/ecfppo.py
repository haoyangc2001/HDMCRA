"""
EC-EFPPO (Energy-Constrained Earliest Feasible PPO) 算法实现。

移植自 Go2HierarchicalMiniCostReachAvoid/rl/EC-EFPPO.py 和
Go2HierarchicalMiniCostReachAvoid/rl/EFPPO_utils.py。

包含两个核心类：
1. EC_EFPPO_Buffer: 经验回放缓冲区，额外存储 energy、energy_consumption、value_reach
2. EC_EFPPO: 训练器，含三个独立优化器（policy、energy critic、reach critic）
"""

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ecfppo_gae import (
    calculate_indexs3,
    calculate_energy_gae,
    calculate_reach_gae,
)
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

@dataclass
class EC_EFPPO_Batch:
    """一个 mini-batch 的训练数据。"""
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values_energy: torch.Tensor
    old_values_reach: torch.Tensor
    advantages_total: torch.Tensor
    targets_energy: torch.Tensor
    targets_reach: torch.Tensor


class EC_EFPPO_Buffer:
    """
    EC-EFPPO 经验回放缓冲区。

    在 ReachAvoidBuffer 的基础上扩展，额外存储：
    - energy: 剩余能量序列 [T+1, N]
    - energy_consumption: 每步能量消耗 [T, N]（作为 reward）
    - value_reach: reach critic 预测值 [T, N]
    """

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device

        obs_dim = obs_shape[0]
        act_dim = action_shape[0]

        # 核心数据
        self.observations = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)

        # Energy value function 预测
        self.values = torch.zeros(horizon, num_envs, device=device)
        # Reach value function 预测
        self.value_reach = torch.zeros(horizon, num_envs, device=device)

        # 环境数据
        self.energy = torch.zeros(horizon + 1, num_envs, device=device)
        self.energy_consumption = torch.zeros(horizon, num_envs, device=device)
        self.g_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.h_values = torch.zeros(horizon + 1, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device, dtype=torch.bool)

        # 优势和目标（由 compute_advantages 填充）
        self.advantages_total = torch.zeros(horizon, num_envs, device=device)
        self.targets_energy = torch.zeros(horizon, num_envs, device=device)
        self.targets_reach = torch.zeros(horizon, num_envs, device=device)

        self.step = 0

    def clear(self) -> None:
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        value_reach: torch.Tensor,
        energy: torch.Tensor,
        energy_consumption: torch.Tensor,
        g_values: torch.Tensor,
        h_values: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_energy: torch.Tensor,
        next_g: torch.Tensor,
        next_h: torch.Tensor,
    ) -> None:
        """
        存储单步 transition 数据。

        Args:
            obs: [N, obs_dim] 当前观测
            actions: [N, act_dim] 采样的动作
            log_probs: [N] 动作的 log 概率
            values: [N] energy critic 预测
            value_reach: [N] reach critic 预测
            energy: [N] 当前剩余能量
            energy_consumption: [N] 本步能量消耗
            g_values: [N] 当前 reach 值（g 值）
            h_values: [N] 当前安全约束值（h 值）
            dones: [N] 环境终止标志
            next_obs: [N, obs_dim] 下一步观测
            next_energy: [N] 下一步剩余能量
            next_g: [N] 下一步 reach 值
            next_h: [N] 下一步安全约束值
        """
        idx = self.step
        self.observations[idx] = obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.values[idx] = values
        self.value_reach[idx] = value_reach
        self.energy[idx] = energy
        self.energy_consumption[idx] = energy_consumption
        self.g_values[idx] = g_values
        self.h_values[idx] = h_values
        self.dones[idx] = dones.bool()
        # 存储下一步的 bootstrap 数据
        self.observations[idx + 1] = next_obs
        self.energy[idx + 1] = next_energy
        self.g_values[idx + 1] = next_g
        self.h_values[idx + 1] = next_h
        self.step += 1

    def compute_advantages(
        self,
        last_values_energy: torch.Tensor,
        last_values_reach: torch.Tensor,
        gamma_energy: float,
        gamma_reach: float,
        gae_lambda: float,
        gamma_reach_init: float,
    ) -> None:
        """
        核心优势计算。对应 JAX 版 _train 中的优势计算部分。

        执行流程：
        1. 调用 calculate_indexs3 计算 earliest reach index → done 矩阵
        2. 将环境 dones 合并到 done 矩阵
        3. 计算 reach 优势 (advantages_h, targets_h)
        4. 计算 energy 优势 (advantages_V, targets_V)
        5. 计算组合优势 (advantages_total)

        Args:
            last_values_energy: [N] 最后一步的 energy value 预测
            last_values_reach: [N] 最后一步的 reach value 预测
            gamma_energy: energy 折扣因子
            gamma_reach: reach 折扣因子（当前值，含退火）
            gae_lambda: GAE λ
            gamma_reach_init: reach 初始折扣因子（用于组合优势）
        """
        if self.step != self.horizon:
            raise RuntimeError("incomplete rollout stored in buffer")

        # ---- 构造扩展序列（含 bootstrap 终止值） ----
        # reach 序列 [T+1, N]
        reach_append = self.g_values
        # reach value 序列 [T+1, N]
        V_reach_append = torch.cat(
            [self.value_reach, last_values_reach.unsqueeze(0)], dim=0
        )
        # energy 序列 [T+1, N]
        energy_append = self.energy
        # energy value 序列 [T+1, N]
        V_energy_append = torch.cat(
            [self.values, last_values_energy.unsqueeze(0)], dim=0
        )

        # ---- 组合信号（对应 JAX 版） ----
        # V_total = max(V_reach, V_energy - energy)
        V_total_append = torch.maximum(V_reach_append, V_energy_append - energy_append)
        # g_append = max(reach, -energy)
        g_append = torch.maximum(reach_append, -energy_append)

        # ---- Step 1: 计算 earliest reach index 和 done 矩阵 ----
        # reward = energy_consumption, energy = energy_append, T_hs = reach_append
        _, done = calculate_indexs3(
            gamma_energy,
            self.energy_consumption,
            energy_append,
            reach_append,
            last_values_energy,
            last_values_reach,
        )
        # done: [T+1, N]，去掉最后一行（对应 bootstrap）
        # 注意：calculate_indexs3 返回的 done 形状是 [T+1, N]
        # 我们需要前 T 行用于 GAE 计算
        done_for_gae = done[:-1, :].clone()  # [T, N]

        # ---- Step 2: 合并环境 dones ----
        done_for_gae = (done_for_gae.bool() | self.dones).float()

        # ---- Step 3: 计算 reach 优势 ----
        advantages_h, targets_h = calculate_reach_gae(
            gamma_reach, gae_lambda, reach_append, V_reach_append, done_for_gae
        )

        # ---- Step 4: 计算 energy 优势 ----
        advantages_V, targets_V = calculate_energy_gae(
            gamma_energy, gae_lambda, self.energy_consumption, self.values,
            done_for_gae, last_values_energy
        )

        # ---- Step 5: 计算组合优势 ----
        advantages_total, _ = calculate_reach_gae(
            gamma_reach_init, gae_lambda, g_append, V_total_append, done_for_gae
        )

        # ---- 存储 ----
        self.advantages_total.copy_(advantages_total)
        self.targets_energy.copy_(targets_V)
        self.targets_reach.copy_(targets_h)

    def _flat_view(self) -> EC_EFPPO_Batch:
        """将 [T, N, ...] 数据展平为 [T*N, ...]。"""
        obs = self.observations[:-1].reshape(-1, self.observations.size(-1))
        actions = self.actions.reshape(-1, self.actions.size(-1))
        log_probs = self.log_probs.reshape(-1)
        values_energy = self.values.reshape(-1)
        values_reach = self.value_reach.reshape(-1)
        advantages_total = self.advantages_total.reshape(-1)
        targets_energy = self.targets_energy.reshape(-1)
        targets_reach = self.targets_reach.reshape(-1)
        return EC_EFPPO_Batch(
            obs, actions, log_probs, values_energy, values_reach,
            advantages_total, targets_energy, targets_reach,
        )

    def iter_batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[EC_EFPPO_Batch]:
        """
        生成用于多轮 mini-batch 训练的数据批次。

        与 ReachAvoidBuffer.iter_batches 逻辑一致。
        """
        data = self._flat_view()
        batch_size = data.observations.size(0)
        mini_batch_size = batch_size // num_mini_batches

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                idx = indices[start:end]
                yield EC_EFPPO_Batch(
                    data.observations[idx],
                    data.actions[idx],
                    data.old_log_probs[idx],
                    data.old_values_energy[idx],
                    data.old_values_reach[idx],
                    data.advantages_total[idx],
                    data.targets_energy[idx],
                    data.targets_reach[idx],
                )


# ---------------------------------------------------------------------------
# EC-EFPPO Trainer
# ---------------------------------------------------------------------------

class EC_EFPPO:
    """
    EC-EFPPO 训练器。

    三个独立优化器分别更新 actor、energy_critic、reach_critic，
    对应 JAX 版三个独立的 TrainState。
    """

    def __init__(
        self,
        actor_critic: EC_EFPPO_ActorCritic,
        learning_rate: float = 3e-4,
        gamma_energy: float = 1.0,
        gamma_reach_init: float = 0.999,
        gamma_reach_final: float = 0.99999,
        gae_lambda: float = 0.95,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        anneal_entropy: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        # 超参数
        self.learning_rate = learning_rate
        self.gamma_energy = gamma_energy
        self.gamma_reach_init = gamma_reach_init
        self.gamma_reach_final = gamma_reach_final
        self.gae_lambda = gae_lambda
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.anneal_entropy = anneal_entropy

        # 三个独立优化器（对应 JAX 版三个独立 TrainState）
        self.policy_optimizer = optim.Adam(
            self.actor_critic.actor.parameters(), lr=learning_rate
        )
        self.energy_optimizer = optim.Adam(
            self.actor_critic.energy_critic.parameters(), lr=learning_rate
        )
        self.reach_optimizer = optim.Adam(
            self.actor_critic.reach_critic.parameters(), lr=learning_rate
        )

        self.buffer = None

    def init_storage(
        self,
        num_envs: int,
        horizon: int,
        obs_shape,
        action_shape,
    ) -> None:
        """初始化缓冲区。"""
        obs_shape = tuple(obs_shape) if isinstance(obs_shape, (list, tuple)) else (obs_shape,)
        action_shape = tuple(action_shape) if isinstance(action_shape, (list, tuple)) else (action_shape,)
        self.buffer = EC_EFPPO_Buffer(
            num_envs, horizon, obs_shape, action_shape, self.device
        )

    def act(
        self, obs: torch.Tensor, critic_obs: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作并记录 value 和 value_reach。

        Returns:
            actions: [N, act_dim]
            log_probs: [N]
            values_energy: [N] energy critic 预测
            values_reach: [N] reach critic 预测
        """
        with torch.no_grad():
            actions, log_probs, values_energy, values_reach = self.actor_critic.act(
                obs, critic_obs
            )
        return actions, log_probs, values_energy, values_reach

    def update(
        self,
        gamma_reach: float,
        entropy_coef: float = None,
    ) -> Dict[str, float]:
        """
        三路独立 PPO 更新。

        对应 JAX 版 _ecefppo_update。

        Args:
            gamma_reach: 当前 reach gamma（含退火）
            entropy_coef: 当前 entropy 系数（含退火），None 则使用 self.entropy_coef

        Returns:
            loss_dict: 包含 actor_loss, energy_loss, reach_loss, entropy_loss
        """
        assert self.buffer is not None

        if entropy_coef is None:
            entropy_coef = self.entropy_coef

        policy_loss_acc = 0.0
        energy_loss_acc = 0.0
        reach_loss_acc = 0.0
        entropy_loss_acc = 0.0
        batch_count = 0

        for batch in self.buffer.iter_batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            obs_batch = batch.observations
            act_batch = batch.actions
            old_log_probs = batch.old_log_probs
            old_values_energy = batch.old_values_energy
            old_values_reach = batch.old_values_reach
            advantages_total = batch.advantages_total
            targets_energy = batch.targets_energy
            targets_reach = batch.targets_reach

            # ===================== Policy 更新 =====================
            # 前向传播 actor
            self.actor_critic.update_distribution(obs_batch)
            log_probs = self.actor_critic.get_actions_log_prob(act_batch)
            entropy = self.actor_critic.entropy.mean()

            # 重要性采样比率
            ratio = torch.exp(log_probs - old_log_probs)

            # 优势归一化（在 loss 内部进行，对应 JAX 版 _loss_fn_policy）
            gae = (advantages_total - advantages_total.mean()) / (
                advantages_total.std() + 1e-8
            )

            # PPO clip 目标
            loss_actor1 = ratio * gae
            loss_actor2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * gae
            )
            policy_loss = -torch.min(loss_actor1, loss_actor2).mean()

            # 总策略损失 = policy_loss - entropy_coef * entropy
            actor_total_loss = policy_loss - entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            actor_total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(), self.max_grad_norm
            )
            self.policy_optimizer.step()

            # ===================== Energy Critic 更新 =====================
            # 前向传播 energy critic
            values = self.actor_critic.energy_critic(obs_batch).squeeze(-1)

            # Clipped value loss（对应 JAX 版 _loss_fn_energy）
            value_pred_clipped = old_values_energy + torch.clamp(
                values - old_values_energy, -self.clip_param, self.clip_param
            )
            value_losses = (values - targets_energy).pow(2)
            value_losses_clipped = (value_pred_clipped - targets_energy).pow(2)
            energy_loss = (
                0.5 * torch.max(value_losses, value_losses_clipped).mean()
            )
            energy_total_loss = self.value_loss_coef * energy_loss

            self.energy_optimizer.zero_grad()
            energy_total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.energy_critic.parameters(), self.max_grad_norm
            )
            self.energy_optimizer.step()

            # ===================== Reach Critic 更新 =====================
            # 前向传播 reach critic
            values_h = self.actor_critic.reach_critic(obs_batch).squeeze(-1)

            # Clipped value loss（对应 JAX 版 _loss_fn_reach）
            value_pred_clipped_reach = old_values_reach + torch.clamp(
                values_h - old_values_reach, -self.clip_param, self.clip_param
            )
            value_losses_reach = (values_h - targets_reach).pow(2)
            value_losses_clipped_reach = (value_pred_clipped_reach - targets_reach).pow(2)
            reach_loss = (
                0.5
                * torch.max(value_losses_reach, value_losses_clipped_reach).mean()
            )
            reach_total_loss = self.value_loss_coef * reach_loss

            self.reach_optimizer.zero_grad()
            reach_total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.reach_critic.parameters(), self.max_grad_norm
            )
            self.reach_optimizer.step()

            # 累加
            policy_loss_acc += policy_loss.item()
            energy_loss_acc += energy_loss.item()
            reach_loss_acc += reach_loss.item()
            entropy_loss_acc += entropy.item()
            batch_count += 1

        # 清空缓冲区
        self.buffer.clear()

        num_updates = max(batch_count, 1)
        return {
            "actor_loss": policy_loss_acc / num_updates,
            "energy_loss": energy_loss_acc / num_updates,
            "reach_loss": reach_loss_acc / num_updates,
            "entropy_loss": entropy_loss_acc / num_updates,
        }

    @staticmethod
    def compute_gamma_reach(
        gamma_reach_init: float,
        gamma_reach_final: float,
        current_update: int,
        total_updates: int,
    ) -> float:
        """
        γ_reach 退火：从 gamma_reach_init 线性增长到 gamma_reach_final。

        对应 JAX 版：
        gamma_2 = min(gamma_reach_final,
                      gamma_reach_init + (gamma_reach_final - gamma_reach_init)
                      * timestep * 2 / total_timesteps)
        """
        gamma = min(
            gamma_reach_final,
            gamma_reach_init
            + (gamma_reach_final - gamma_reach_init) * current_update * 2 / total_updates,
        )
        return gamma

    @staticmethod
    def compute_entropy_coef(
        entropy_coef_init: float,
        current_update: int,
        total_updates: int,
        anneal: bool = True,
    ) -> float:
        """
        Entropy 退火：从 entropy_coef_init 线性衰减到 0。

        对应 JAX 版：
        ent = ent_coef * (total_timesteps - timestep) / total_timesteps
        """
        if not anneal:
            return entropy_coef_init
        return entropy_coef_init * (total_updates - current_update) / total_updates
