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
    """
    计算 Earliest Reach Index 和 done 矩阵。
    
    移植自 JAX 版 calculate_indexs3 (gae.py 第 334-378 行)。
    反向扫描轨迹，维护 Vs_row（能量累计值）和 Vhs_row（reach 值），
    在每个时间步计算 V_total = max(Vs - energy, Vhs)，然后找到使组合价值
    最小的时间步索引作为 done 标志。
    
    Args:
        gamma: 折扣因子
        reward: [T, nh] 能量消耗奖励
        energy: [T+1, nh] 能量值序列（包含终止状态）
        T_hs: [T+1, nh] reach 值序列（包含终止状态）
        last_value: [nh] 最后时间步的 energy value function 预测
        last_value_reach: [nh] 最后时间步的 reach value function 预测
    
    Returns:
        indexs: [T, nh] 每个时间步的 earliest reach index
        done: [T+1, nh] done 矩阵（1.0 表示该位置是 earliest reach）
    """
    device = reward.device
    dtype = reward.dtype
    T, nh = reward.shape
    Tp1 = T + 1

    # 初始化 carry
    Vs_row = torch.full((T, nh), float('inf'), device=device, dtype=dtype)
    Vhs_row = torch.full((T, nh), float('inf'), device=device, dtype=dtype)
    done = torch.zeros(Tp1, nh, device=device, dtype=dtype)
    mask_1 = torch.full((T, 1), float('inf'), device=device, dtype=dtype)

    indexs = torch.zeros(T, nh, device=device, dtype=torch.long)

    # 反向迭代 (对应 JAX lax.scan reverse=True)
    for ii in range(T - 1, -1, -1):
        # Vs_row = mask_1 * (reward[ii] + gamma * next_Vs_row)
        # mask_1 形状 (T, 1)，broadcast 到 (T, nh)
        Vs_row = mask_1 * (reward[ii] + gamma * Vs_row)
        Vs_row[ii] = 0.0  # 当前步能量消耗为 0

        # Vhs_row[ii] = reach[ii]
        Vhs_row = Vhs_row.clone()
        Vhs_row[ii] = T_hs[ii]

        # V_total = max(Vs - energy, Vhs)
        # energy[-ii-1] 对应 JAX 版
        energy_idx = T - ii - 1
        V_total = torch.maximum(Vs_row - energy[energy_idx], Vhs_row)

        # 反转 V_total
        V_total_rev = V_total.flip(0)

        # V_next = max(gamma^ii * last_value + V_total[-1] - energy[-ii-1], last_value_reach)
        V_next = torch.maximum(
            (gamma ** ii) * last_value + V_total_rev[-1] - energy[energy_idx],
            last_value_reach
        )

        # V_total_1 = concatenate(V_total_rev, V_next.unsqueeze(0))
        V_total_1 = torch.cat([V_total_rev, V_next.unsqueeze(0)], dim=0)

        # argmin
        index_1 = torch.argmin(V_total_1, dim=0)

        # done[index_1, arange(nh)] = 1.0
        done[index_1, torch.arange(nh, device=device)] = 1.0

        # 保存 indexs
        indexs[ii] = index_1

        # 更新 mask_1: roll right, set [0] = 1.0
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
    """
    计算单步 advantage（带 done/next_done 双重 mask）。
    
    移植自 JAX 版 calculate_advantage2 (gae.py 第 394-404 行)。
    
    Args:
        gae: [nh] 当前 GAE 累积值
        next_value: [nh] 下一步的 value
        gamma: 折扣因子
        lam: GAE lambda
        reward: [nh] 当前步奖励
        value: [nh] 当前步 value
        done: [nh] 当前步 done 标志
        next_done: [nh] 下一步 done 标志
    
    Returns:
        (gae, value, gamma, lam): 更新后的 carry
        gae: 当前步的 advantage
    """
    # delta = (reward + gamma * next_value * (1 - next_done)) * (1 - done) - value
    delta = (reward + gamma * next_value * (1 - next_done)) * (1 - done) - value
    # gae = delta + gamma * lam * (1 - next_done) * (1 - done) * gae
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
    """
    计算 Energy Value Function 的 GAE（带 done/next_done 双重 mask）。
    
    移植自 JAX 版 calculate_gae2 (gae.py 第 419-432 行)。
    用于计算 energy value function 的优势函数，使用 done 和 next_done 双重 mask
    控制回报传播。
    
    Args:
        gamma: 折扣因子
        gae_lambda: GAE lambda
        rewards: [T, nh] 奖励序列
        values: [T, nh] value 预测序列
        done: [T, nh] done 标志
        last_value: [nh] 最后时间步的 value 预测
    
    Returns:
        advantages: [T, nh] 优势估计
        targets: [T, nh] value 目标
    """
    T, nh = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    
    # 计算 next_done: roll done left by 1, last element = second-to-last
    next_done = torch.roll(done, shifts=-1, dims=0)
    next_done[-1] = next_done[-2]
    
    # 反向迭代计算 GAE
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
    """
    计算 Reach Value Function 的 GAE。
    
    移植自 JAX 版 calculate_gae_reach4 (gae.py 第 195-247 行)。
    
    Args:
        gamma: 折扣因子
        gae_lambda: GAE lambda
        h_seq: [T+1, nh] reach 值序列（包含终止状态）
        Vhs_seq: [T+1, nh] reach value function 序列（包含终止状态）
        done: [T+1, nh] done 标志（整数类型，1 表示 done）
    
    Returns:
        advantages: [T, nh] 优势估计
        q_targets: [T, nh] Q 值目标
    """
    device = h_seq.device
    dtype = h_seq.dtype
    Tp1, nh = h_seq.shape
    T = Tp1 - 1
    
    # 确保 done 是整数类型
    done = done.to(torch.long)
    
    # 初始化
    lam_ratio = gae_lambda / max(1.0 - gae_lambda, 1e-6)
    gae_coeffs = torch.zeros(T + 1, nh, device=device, dtype=dtype)
    value_table = torch.zeros(T + 1, nh, device=device, dtype=dtype)
    value_table[0] = Vhs_seq[T]  # 初始化为终止状态的 value
    pre_done = torch.zeros(nh, device=device, dtype=dtype)
    
    q_targets = torch.zeros(T, nh, device=device, dtype=dtype)
    
    # 反向迭代
    for ii in range(T - 1, -1, -1):
        done_row = done[ii].to(dtype)
        
        # 更新 GAE 系数
        rolled = torch.roll(gae_coeffs, shifts=1, dims=0)
        gae_coeffs = (
            rolled * gae_lambda * (1.0 - pre_done)
            + rolled * lam_ratio * pre_done
        ) * (1.0 - done_row)
        gae_coeffs[0] = 1.0
        
        # 时间步掩码
        mask = (torch.arange(T + 1, device=device) < ii + 1).to(dtype).unsqueeze(1)
        
        # done_row_processed: done * inf，但 nan 处理为 0
        done_inf = done_row * float('inf')
        done_inf = torch.where(torch.isnan(done_inf), torch.zeros_like(done_inf), done_inf)
        
        # DP for Vh
        # disc_to_h = (1-gamma)*h + gamma*(Vhs + done_inf)
        disc_to_h = (1.0 - gamma) * h_seq[ii] + gamma * (value_table[0] + done_inf)
        
        # Vhs_row = min(h, disc_to_h)
        Vhs_row_single = torch.minimum(h_seq[ii], disc_to_h)
        
        # 扩展到 [T+1, nh] 形状并应用掩码
        Vhs_row = Vhs_row_single.unsqueeze(0).expand(T + 1, -1) * mask
        
        # 归一化 GAE 系数并计算 Q 值
        coeff_sum = gae_coeffs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        norm_coeffs = gae_coeffs / coeff_sum
        Qhs_GAE = (Vhs_row * norm_coeffs).sum(dim=0)
        
        q_targets[ii] = Qhs_GAE
        
        # 更新 value_table: roll right, set [0] = Vhs_seq[ii+1]
        Vhs_row_rolled = torch.roll(Vhs_row, shifts=1, dims=0)
        Vhs_row_rolled[0] = Vhs_seq[ii + 1]
        value_table = Vhs_row_rolled
        
        pre_done = done_row
    
    advantages = q_targets - Vhs_seq[:-1]
    return advantages, q_targets


# 为了兼容性，也提供 JAX 版函数名的别名
calculate_indexs = calculate_indexs3
calculate_gae_reach = calculate_reach_gae
