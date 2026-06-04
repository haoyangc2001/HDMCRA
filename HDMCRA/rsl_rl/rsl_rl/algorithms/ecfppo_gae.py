"""
EC-EFPPO 核心 GAE 算法实现
移植自 Go2HierarchicalMiniCostReachAvoid/rl/gae.py

包含三个核心函数：
1. calculate_indexs3: 计算 earliest reach index 和 done 矩阵
2. calculate_energy_gae: 计算 energy value function 的 GAE (对应 JAX calculate_gae2)
3. calculate_reach_gae: 计算 reach value function 的 GAE (对应 JAX calculate_gae_reach4)
"""

import torch
from typing import Tuple


def calculate_indexs3(
    gamma: float,
    reward: torch.Tensor,
    energy: torch.Tensor,
    T_hs: torch.Tensor,
    last_value: torch.Tensor,
    last_value_reach: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch port of JAX calculate_indexs3."""
    device = reward.device
    dtype = reward.dtype
    T, nh = reward.shape
    Tp1 = T + 1

    Vs_row = torch.full((T, nh), float("inf"), device=device, dtype=dtype)
    Vhs_row = torch.full((T, nh), float("inf"), device=device, dtype=dtype)
    done = torch.zeros(Tp1, nh, device=device, dtype=dtype)
    mask_1 = torch.full((T, 1), float("inf"), device=device, dtype=dtype)
    indexs = torch.zeros(T, nh, device=device, dtype=torch.long)

    for ii in range(T - 1, -1, -1):
        Vs_row = mask_1 * (reward[ii] + gamma * Vs_row)
        Vs_row[ii] = 0.0

        Vhs_row = Vhs_row.clone()
        Vhs_row[ii] = T_hs[ii]

        energy_idx = T - ii
        V_total = torch.maximum(Vs_row - energy[energy_idx], Vhs_row).flip(0)
        V_next = torch.maximum(
            (gamma ** ii) * last_value + V_total[-1] - energy[energy_idx],
            last_value_reach,
        )
        V_total_1 = torch.cat([V_total, V_next.unsqueeze(0)], dim=0)

        index_1 = torch.argmin(V_total_1, dim=0)
        done[index_1, torch.arange(nh, device=device)] = 1.0
        indexs[ii] = index_1

        mask_1 = torch.roll(mask_1, shifts=1, dims=0)
        mask_1[0] = 1.0

    return indexs, done


def calculate_advantage2(
    gae: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    next_done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """PyTorch port of JAX calculate_advantage2."""
    delta = (reward + gamma * next_value * (1 - next_done)) * (1 - done) - value
    gae = delta + gamma * lam * (1 - next_done) * (1 - done) * gae
    return gae, value, gamma, lam


def calculate_energy_gae(
    gamma: float,
    gae_lambda: float,
    rewards: torch.Tensor,
    values: torch.Tensor,
    done: torch.Tensor,
    last_value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch port of JAX calculate_gae2."""
    T, nh = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    next_done = torch.roll(done, shifts=-1, dims=0)
    if T > 1:
        next_done[-1] = next_done[-2]

    gae = torch.zeros(nh, device=device, dtype=dtype)
    next_value = last_value
    advantages = torch.zeros(T, nh, device=device, dtype=dtype)

    for t in range(T - 1, -1, -1):
        gae, next_value, _, _ = calculate_advantage2(
            gae, next_value, gamma, gae_lambda,
            rewards[t], values[t], done[t], next_done[t]
        )
        advantages[t] = gae
        next_value = values[t]

    targets = advantages + values
    return advantages, targets


def calculate_reach_gae(
    gamma: float,
    gae_lambda: float,
    h_seq: torch.Tensor,
    Vhs_seq: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch port of JAX calculate_gae_reach4."""
    device = h_seq.device
    dtype = h_seq.dtype
    Tp1, nh = h_seq.shape
    T = Tp1 - 1

    done = done.to(torch.long)
    lam_ratio = gae_lambda / max(1.0 - gae_lambda, 1e-6)
    gae_coeffs = torch.zeros(T + 1, nh, device=device, dtype=dtype)
    value_table = torch.zeros(T + 1, nh, device=device, dtype=dtype)
    value_table[0] = Vhs_seq[T]
    pre_done = torch.zeros(nh, device=device, dtype=dtype)
    q_targets = torch.zeros(T, nh, device=device, dtype=dtype)

    for ii in range(T - 1, -1, -1):
        done_row = done[ii].to(dtype).unsqueeze(0)
        pre_done_row = pre_done.unsqueeze(0)

        rolled = torch.roll(gae_coeffs, shifts=1, dims=0)
        gae_coeffs = (
            rolled * gae_lambda * (1.0 - pre_done_row)
            + rolled * lam_ratio * pre_done_row
        ) * (1.0 - done_row)
        gae_coeffs[0] = 1.0

        mask = (torch.arange(T + 1, device=device) < ii + 1).to(dtype).unsqueeze(1)

        done_inf = done_row * float("inf")
        done_inf = torch.where(torch.isnan(done_inf), torch.zeros_like(done_inf), done_inf)
        disc_to_h = (1.0 - gamma) * h_seq[ii].unsqueeze(0) + gamma * (value_table + done_inf)
        Vhs_row = torch.minimum(h_seq[ii].unsqueeze(0), disc_to_h)
        Vhs_row = mask * Vhs_row

        normed_gae_coeffs = gae_coeffs / gae_coeffs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        q_targets[ii] = (Vhs_row * normed_gae_coeffs).sum(dim=0)

        Vhs_row = torch.roll(Vhs_row, shifts=1, dims=0)
        Vhs_row[0] = Vhs_seq[ii + 1]
        value_table = Vhs_row
        pre_done = done_row.squeeze(0)

    return q_targets - Vhs_seq[:-1], q_targets


calculate_indexs = calculate_indexs3
calculate_gae_reach = calculate_reach_gae
