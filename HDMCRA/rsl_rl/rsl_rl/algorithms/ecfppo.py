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

from rsl_rl.utils.running_mean_std import RunningMeanStd
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
        reach_value_clip: float = None,
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device
        self.reach_value_clip = reach_value_clip

        obs_dim = obs_shape[0]
        act_dim = action_shape[0]

        # 核心数据
        self.observations = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        # D005 诊断：记录 actor 分布均值，用于区分动作贴边来自均值还是采样噪声。
        self.action_mean = torch.zeros(horizon, num_envs, act_dim, device=device)
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

        self.debug_stats: Dict[str, float] = {}
        self.step = 0

    def clear(self) -> None:
        self.step = 0
        self.debug_stats = {}

    @staticmethod
    def _stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
        values = tensor.detach().float()
        return {
            f"{prefix}_min": values.min().item(),
            f"{prefix}_max": values.max().item(),
            f"{prefix}_mean": values.mean().item(),
            f"{prefix}_std": values.std(unbiased=False).item(),
        }

    @staticmethod
    def _masked_stats(prefix: str, tensor: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        values = tensor.detach().float()[mask.bool()]
        stats = {f"{prefix}_count": float(values.numel())}
        if values.numel() == 0:
            stats.update({
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
                f"{prefix}_mean": float("nan"),
                f"{prefix}_std": float("nan"),
            })
        else:
            stats.update({
                f"{prefix}_min": values.min().item(),
                f"{prefix}_max": values.max().item(),
                f"{prefix}_mean": values.mean().item(),
                f"{prefix}_std": values.std(unbiased=False).item(),
            })
        return stats

    @staticmethod
    def _dim_stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
        """按动作维度记录均值，避免总均值掩盖某一维异常。"""
        values = tensor.detach().float()
        if values.dim() < 3:
            return {}
        flat = values.reshape(-1, values.size(-1))
        return {
            f"{prefix}_dim{dim}": flat[:, dim].mean().item()
            for dim in range(flat.size(-1))
        }

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
        action_mean: torch.Tensor = None,
    ) -> None:
        """
        存储单步 transition 数据。

        Args:
            obs: [N, obs_dim] 当前高层决策前的状态 s_t 观测
            actions: [N, act_dim] 在 s_t 采样的高层动作 a_t
            action_mean: [N, act_dim] actor 在 s_t 输出的动作分布均值，用于诊断动作贴边来源
            log_probs: [N] a_t 在 s_t 下的 log 概率
            values: [N] energy critic 在 s_t 的预测
            value_reach: [N] reach critic 在 s_t 的预测
            energy: [N] 状态 s_t 的剩余能量
            energy_consumption: [N] 执行动作 a_t 后，从 s_t -> s_{t+1} 的本步能量消耗
            g_values: [N] 状态 s_t 的 reach 值（g 值）
            h_values: [N] 状态 s_t 的安全约束值（h 值）
            dones: [N] 执行动作 a_t 后到达 s_{t+1} 时的环境终止标志
            next_obs: [N, obs_dim] 下一步状态 s_{t+1} 观测
            next_energy: [N] 状态 s_{t+1} 的剩余能量
            next_g: [N] 状态 s_{t+1} 的 reach 值
            next_h: [N] 状态 s_{t+1} 的安全约束值
        """
        idx = self.step
        self.observations[idx] = obs
        self.actions[idx] = actions
        self.action_mean[idx] = actions if action_mean is None else action_mean
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
        energy_target_rms=None,
    ) -> None:
        """
        核心优势计算。对应 JAX 版 _train 中的优势计算部分。

        执行流程：
        1. 缓冲区中 [t] 行统一表示决策前状态 s_t，[t+1] 行表示执行动作后的状态 s_{t+1}
        2. 调用 calculate_indexs3 计算 earliest reach index → done 矩阵
        3. 将环境 dones 合并到 done 矩阵
        4. 计算 reach 优势 (advantages_h, targets_h)
        5. 计算 energy 优势 (advantages_V, targets_V)
        6. 计算组合优势 (advantages_total)

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
        if self.reach_value_clip is not None:
            clip = float(self.reach_value_clip)
            V_reach_for_bootstrap = V_reach_append.clamp(-clip, clip)
            last_values_reach_for_bootstrap = last_values_reach.clamp(-clip, clip)
        else:
            V_reach_for_bootstrap = V_reach_append
            last_values_reach_for_bootstrap = last_values_reach
        # energy 序列 [T+1, N]
        energy_append = self.energy
        # energy value 序列 [T+1, N]
        V_energy_append = torch.cat(
            [self.values, last_values_energy.unsqueeze(0)], dim=0
        )

        # ---- 组合信号（对应 JAX 版） ----
        # V_total = max(V_reach, V_energy - energy)
        V_total_append = torch.maximum(V_reach_for_bootstrap, V_energy_append - energy_append)
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
            last_values_reach_for_bootstrap,
        )
        # done: [T+1, N]，去掉最后一行（对应 bootstrap）
        # 注意：calculate_indexs3 返回的 done 形状是 [T+1, N]
        # 我们需要前 T 行用于 GAE 计算
        done_for_gae = done[:-1, :].clone()  # [T, N]

        # ---- Step 2: 合并环境 dones ----
        done_for_gae = (done_for_gae.bool() | self.dones).float()

        # ---- Step 3: 计算 reach 优势 ----
        advantages_h, targets_h = calculate_reach_gae(
            gamma_reach, gae_lambda, reach_append, V_reach_for_bootstrap, done_for_gae
        )

        # ---- Step 4: 计算 energy 优势 ----
        advantages_V, targets_V = calculate_energy_gae(
            gamma_energy, gae_lambda, self.energy_consumption, self.values,
            done_for_gae, last_values_energy
        )

        # ---- Step 5: 计算组合优势 ----
        # 这里刻意沿用 JAX 参考实现的口径：combined advantage 使用 gamma_reach_init，
        # 而不是当前退火后的 gamma_reach。该设计用于让策略更新信号保持一个固定的
        # 初始 reach 折扣语义；如需改成当前 gamma_reach，必须同步更新测试和文档。
        advantages_total, _ = calculate_reach_gae(
            gamma_reach_init, gae_lambda, g_append, V_total_append, done_for_gae
        )

        # ---- 归一化 energy critic 目标值 ----
        if energy_target_rms is not None:
            # 更新 energy_target_rms 统计量
            energy_target_rms.update(targets_V.reshape(-1, 1))
            # 归一化 targets_V
            targets_V_normalized = energy_target_rms.normalize(
                targets_V.reshape(-1, 1), clip_range=10.0
            ).reshape(targets_V.shape)
        else:
            targets_V_normalized = targets_V

        # ---- 存储 ----
        self.advantages_total.copy_(advantages_total)
        self.targets_energy.copy_(targets_V_normalized)  # 存储归一化后的目标
        self.targets_reach.copy_(targets_h)

        # 诊断统计用于定位 critic/target/advantage 量级问题。
        self.debug_stats = {}
        self.debug_stats.update(self._stats("values_reach", self.value_reach))
        if self.reach_value_clip is not None:
            clip = float(self.reach_value_clip)
            self.debug_stats["reach_value_clip"] = clip
            self.debug_stats["reach_value_clip_ratio"] = (
                (V_reach_append.detach().abs() > clip).float().mean().item()
            )
        else:
            self.debug_stats["reach_value_clip"] = float("nan")
            self.debug_stats["reach_value_clip_ratio"] = 0.0
        self.debug_stats.update(self._stats("targets_reach", self.targets_reach))
        self.debug_stats.update(self._stats("values_energy", self.values))
        self.debug_stats.update(self._stats("targets_energy", self.targets_energy))
        self.debug_stats.update(self._stats("advantages_total", self.advantages_total))
        self.debug_stats.update(self._stats("g_values", self.g_values))
        self.debug_stats.update(self._stats("h_values", self.h_values))
        self.debug_stats["done_for_gae_mean"] = done_for_gae.float().mean().item()

        done_mask = done_for_gae.bool()
        open_mask = ~done_mask
        self.debug_stats.update(self._masked_stats("targets_reach_done", self.targets_reach, done_mask))
        self.debug_stats.update(self._masked_stats("targets_reach_open", self.targets_reach, open_mask))
        self.debug_stats.update(self._masked_stats("values_reach_done", self.value_reach, done_mask))
        self.debug_stats.update(self._masked_stats("values_reach_open", self.value_reach, open_mask))
        self.debug_stats.update(self._masked_stats("advantages_total_done", self.advantages_total, done_mask))
        self.debug_stats.update(self._masked_stats("advantages_total_open", self.advantages_total, open_mask))
        done_env_mean = done_for_gae.float().mean(dim=0)
        self.debug_stats.update(self._stats("done_env_mean", done_env_mean))
        self.debug_stats["done_for_gae_open_ratio"] = open_mask.float().mean().item()

        flat_min_idx = torch.argmin(self.targets_reach.reshape(-1))
        min_t = int(flat_min_idx // self.num_envs)
        min_env = int(flat_min_idx % self.num_envs)
        self.debug_stats["targets_reach_min_t"] = float(min_t)
        self.debug_stats["targets_reach_min_done"] = done_for_gae[min_t, min_env].float().item()
        self.debug_stats["targets_reach_min_value_reach"] = (
            self.value_reach[min_t, min_env].detach().float().item()
        )
        self.debug_stats["targets_reach_min_next_value_reach"] = (
            V_reach_for_bootstrap[min_t + 1, min_env].detach().float().item()
        )
        self.debug_stats["targets_reach_min_g"] = self.g_values[min_t, min_env].detach().float().item()
        self.debug_stats["targets_reach_min_h"] = self.h_values[min_t, min_env].detach().float().item()
        self.debug_stats["targets_reach_min_energy"] = self.energy[min_t, min_env].detach().float().item()

        energy_min_mask = energy_append <= energy_append.min() + 1e-6
        self.debug_stats["energy_min_ratio"] = energy_min_mask.float().mean().item()
        self.debug_stats["energy_negative_ratio"] = (energy_append < 0).float().mean().item()
        self.debug_stats.update(self._stats("energy_consumption", self.energy_consumption))
        self.debug_stats.update(self._stats("init_energy", energy_append[0]))

        # 只做诊断：记录每个环境第一次触到能量下界的时间步。
        first_energy_min_step = torch.argmax(energy_min_mask.long(), dim=0)
        never_min = ~energy_min_mask.any(dim=0)
        first_energy_min_step = torch.where(
            never_min,
            torch.full_like(first_energy_min_step, energy_append.shape[0]),
            first_energy_min_step,
        ).float()
        self.debug_stats.update(self._stats("first_energy_min_step", first_energy_min_step))

        action_abs = self.actions.detach().abs()
        action_mean_abs = self.action_mean.detach().abs()
        clipped_action_abs = self.actions.detach().clamp(-1.0, 1.0).abs()
        action_clip_mask = action_abs >= 1.0 - 1e-6
        action_mean_clip_mask = action_mean_abs >= 1.0 - 1e-6
        self.debug_stats.update(self._stats("action_abs", action_abs))
        self.debug_stats.update(self._stats("action_mean_abs", action_mean_abs))
        self.debug_stats.update(self._stats("clipped_action_abs", clipped_action_abs))
        self.debug_stats.update(self._dim_stats("action_mean_abs_mean", action_mean_abs))
        self.debug_stats.update(self._dim_stats("action_mean_clip_ratio", action_mean_clip_mask.float()))
        self.debug_stats.update(self._dim_stats("clipped_action_abs_mean", clipped_action_abs))
        self.debug_stats["action_clip_ratio"] = action_clip_mask.float().mean().item()
        self.debug_stats["action_mean_clip_ratio"] = action_mean_clip_mask.float().mean().item()

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
        policy_learning_rate: float = None,
        energy_learning_rate: float = None,
        reach_learning_rate: float = None,
        gamma_energy: float = 1.0,
        gamma_reach_init: float = 0.999,
        gamma_reach_final: float = 0.99999,
        gae_lambda: float = 0.95,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        actor_mean_bound: float = 1.0,
        actor_mean_bound_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        max_grad_norm_energy: float = None,
        reach_value_clip: float = None,
        anneal_entropy: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)

        # 超参数
        self.learning_rate = learning_rate
        self.policy_learning_rate = (
            learning_rate if policy_learning_rate is None else policy_learning_rate
        )
        self.energy_learning_rate = (
            learning_rate if energy_learning_rate is None else energy_learning_rate
        )
        self.reach_learning_rate = (
            learning_rate if reach_learning_rate is None else reach_learning_rate
        )
        self.gamma_energy = gamma_energy
        self.gamma_reach_init = gamma_reach_init
        self.gamma_reach_final = gamma_reach_final
        self.gae_lambda = gae_lambda
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_mean_bound = float(actor_mean_bound)
        self.actor_mean_bound_coef = float(actor_mean_bound_coef)
        self.max_grad_norm = max_grad_norm
        # Energy critic 使用更严格的梯度裁剪，防止 loss 爆炸
        # 如果未指定，默认使用 max_grad_norm 的 1/5
        self.max_grad_norm_energy = max_grad_norm_energy if max_grad_norm_energy is not None else max_grad_norm / 5.0
        self.reach_value_clip = reach_value_clip
        self.anneal_entropy = anneal_entropy

        # 三个独立优化器（对应 JAX 版三个独立 TrainState）
        # policy optimizer 必须同时管理 actor MLP 和动作分布 log_std；否则 entropy/std 不会更新。
        self.policy_params = list(self.actor_critic.actor.parameters()) + [self.actor_critic.log_std]
        self.policy_optimizer = optim.Adam(
            self.policy_params, lr=self.policy_learning_rate
        )
        self.energy_optimizer = optim.Adam(
            self.actor_critic.energy_critic.parameters(), lr=self.energy_learning_rate
        )
        self.reach_optimizer = optim.Adam(
            self.actor_critic.reach_critic.parameters(), lr=self.reach_learning_rate
        )

        # Energy critic 目标值归一化器
        # 用于归一化 energy critic 的目标值（累积能量消耗）
        # 解决输入（归一化观测 [-10, 10]）与目标（原始累积消耗 [0, 2000+]）的尺度不匹配
        self.energy_target_rms = RunningMeanStd(shape=(), device=self.device)

        self.buffer = None

    def _actor_mean_bound_loss(self, action_mean: torch.Tensor) -> torch.Tensor:
        """惩罚 actor mean 超出环境执行动作边界的部分。"""
        return torch.relu(action_mean.abs() - self.actor_mean_bound).pow(2).mean()

    @staticmethod
    def _policy_gae_from_advantages(advantages_total: torch.Tensor) -> torch.Tensor:
        """将 cost-like reach-avoid advantage 转成 PPO reward-max loss 使用的方向。"""
        normalized = (advantages_total - advantages_total.mean()) / (
            advantages_total.std() + 1e-8
        )
        # advantages_total 来自 g_append=max(reach, -energy)，语义是越小越好。
        return -normalized

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
            num_envs, horizon, obs_shape, action_shape, self.device,
            reach_value_clip=self.reach_value_clip,
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
        mean_bound_loss_acc = 0.0
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

            # advantages_total 是 reach-avoid/cost-like 信号，越小越好；
            # policy loss 仍按 PPO reward-max 形式写，因此这里先取反。
            gae = self._policy_gae_from_advantages(advantages_total)

            # PPO clip 目标
            loss_actor1 = ratio * gae
            loss_actor2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * gae
            )
            policy_loss = -torch.min(loss_actor1, loss_actor2).mean()

            # 总策略损失 = PPO policy loss - entropy bonus + actor mean 边界正则。
            # 正则只惩罚均值越过环境执行边界的部分，不改变采样分布定义。
            action_mean = self.actor_critic.action_mean
            mean_bound_loss = self._actor_mean_bound_loss(action_mean)
            actor_total_loss = (
                policy_loss
                - entropy_coef * entropy
                + self.actor_mean_bound_coef * mean_bound_loss
            )

            self.policy_optimizer.zero_grad()
            actor_total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_params, self.max_grad_norm
            )
            self.policy_optimizer.step()
            self.actor_critic.clamp_log_std_()

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
            # Energy critic 使用更严格的梯度裁剪，防止 loss 爆炸
            nn.utils.clip_grad_norm_(
                self.actor_critic.energy_critic.parameters(), self.max_grad_norm_energy
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
            mean_bound_loss_acc += mean_bound_loss.item()
            batch_count += 1

        # 清空缓冲区
        self.buffer.clear()

        num_updates = max(batch_count, 1)
        return {
            "actor_loss": policy_loss_acc / num_updates,
            "energy_loss": energy_loss_acc / num_updates,
            "reach_loss": reach_loss_acc / num_updates,
            "entropy_loss": entropy_loss_acc / num_updates,
            "mean_bound_loss": mean_bound_loss_acc / num_updates,
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

    def get_energy_target_rms_state(self) -> dict:
        """获取 energy_target_rms 状态（用于保存 checkpoint）。"""
        return self.energy_target_rms.state_dict()

    def set_energy_target_rms_state(self, state: dict) -> None:
        """恢复 energy_target_rms 状态（用于加载 checkpoint）。"""
        if state:
            self.energy_target_rms.load_state_dict(state)
