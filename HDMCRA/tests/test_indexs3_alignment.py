#!/usr/bin/env python3
"""
严格对拍测试：验证 PyTorch calculate_indexs3 与 JAX 参考实现的等价性。

使用小规模数据 (T=5, nh=2)，逐行追踪每个变量的值，
对比 PyTorch 实现和 JAX 参考实现的行为。

运行：conda run -n hdmcr python tests/test_indexs3_alignment.py
"""

import torch
import numpy as np


def calculate_indexs3_pytorch(gamma, reward, energy, T_hs, last_value, last_value_reach):
    """PyTorch 版本（使用修复后的 energy_idx = T - ii）"""
    device = reward.device
    dtype = reward.dtype
    T, nh = reward.shape
    Tp1 = T + 1

    Vs_row = torch.full((T, nh), float("inf"), device=device, dtype=dtype)
    Vhs_row = torch.full((T, nh), float("inf"), device=device, dtype=dtype)
    done = torch.zeros(Tp1, nh, device=device, dtype=dtype)
    mask_1 = torch.full((T, 1), float("inf"), device=device, dtype=dtype)
    indexs = torch.zeros(T, nh, device=device, dtype=torch.long)

    trace = []

    for ii in range(T - 1, -1, -1):
        Vs_row = mask_1 * (reward[ii] + gamma * Vs_row)
        Vs_row[ii] = 0.0

        Vhs_row = Vhs_row.clone()
        Vhs_row[ii] = T_hs[ii]

        # 修复：使用 energy_idx = T - ii 而非 T - ii - 1
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

        trace.append({
            'ii': ii,
            'energy_idx': energy_idx,
            'Vs_row': Vs_row.clone(),
            'Vhs_row': Vhs_row.clone(),
            'energy_used': energy[energy_idx].clone(),
            'V_total': V_total.clone(),
            'V_next': V_next.clone(),
            'V_total_1': V_total_1.clone(),
            'index_1': index_1.clone(),
            'done_sum': done.sum().item(),
        })

        mask_1 = torch.roll(mask_1, shifts=1, dims=0)
        mask_1[0] = 1.0

    return indexs, done, trace


def calculate_indexs3_jax_manual(gamma, reward, energy, T_hs, last_value, last_value_reach):
    """
    手动模拟 JAX 版本的行为。

    JAX 版本关键代码：
    ```python
    Tp1, nh = T_hs.shape
    T = Tp1 - 1

    def loop(carry, inp):
        ii, reach, reward = inp
        (next_Vs_row, next_Vhs_row, next_mask_1, done) = carry

        Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        Vs_row = Vs_row.at[ii, :].set(0)

        Vhs_row = next_Vhs_row.at[ii, :].set(reach)

        V_total = jnp.maximum(Vs_row - energy[-ii-1, :], Vhs_row)[::-1]
        ...

    inps = (ts, T_hs[:-1], reward)
    ```
    """
    # 转换为 numpy 模拟 JAX
    reward_np = reward.numpy() if isinstance(reward, torch.Tensor) else np.array(reward)  # [T, nh]
    energy_np = energy.numpy() if isinstance(energy, torch.Tensor) else np.array(energy)  # [T+1, nh]
    T_hs_np = T_hs.numpy() if isinstance(T_hs, torch.Tensor) else np.array(T_hs)  # [T+1, nh]
    last_value_np = last_value.numpy() if isinstance(last_value, torch.Tensor) else np.array(last_value)  # [nh]
    last_value_reach_np = last_value_reach.numpy() if isinstance(last_value_reach, torch.Tensor) else np.array(last_value_reach)  # [nh]

    T, nh = reward_np.shape
    Tp1 = T + 1

    # 初始化（与 JAX 一致）
    Vs_row = np.full((T, nh), np.inf)
    Vhs_row = np.full((T, nh), np.inf)
    done = np.zeros((Tp1, nh))
    mask_1 = np.full((T, 1), np.inf)

    trace = []

    for idx, ii in enumerate(range(T - 1, -1, -1)):
        # reach 和 reward 的索引
        reach_val = T_hs_np[ii]  # [nh]
        reward_step = reward_np[ii]  # [nh]

        # Vs_row 计算
        # JAX: Vs_row = next_mask_1 * (reward + gamma * next_Vs_row)
        # mask_1: [T, 1], reward_step: [nh], Vs_row: [T, nh]
        # reward_step + gamma * Vs_row: [T, nh] (broadcasting)
        Vs_row = mask_1 * (reward_step + gamma * Vs_row)
        # JAX: Vs_row = Vs_row.at[ii, :].set(0)
        Vs_row[ii, :] = 0.0

        # Vhs_row 计算
        # JAX: Vhs_row = next_Vhs_row.at[ii, :].set(reach)
        Vhs_row = Vhs_row.copy()
        Vhs_row[ii, :] = reach_val

        # energy 索引
        # JAX: energy[-ii-1, :]
        # 当 ii=T-1: energy[-T, :] = energy[0, :]
        # 当 ii=T-2: energy[-(T-1), :] = energy[1, :]
        # 当 ii=0: energy[-1, :] = energy[T, :]
        energy_idx_jax = T - ii  # JAX 的 energy 索引（从 0 到 T）
        energy_idx_pytorch = T - ii - 1  # PyTorch 的 energy 索引（从 0 到 T-1）

        # V_total 计算
        # JAX: V_total = jnp.maximum(Vs_row - energy[-ii-1, :], Vhs_row)[::-1]
        V_total_jax = np.maximum(Vs_row - energy_np[energy_idx_jax], Vhs_row)[::-1]
        # PyTorch: V_total = torch.maximum(Vs_row - energy[energy_idx], Vhs_row).flip(0)
        V_total_pytorch = np.maximum(Vs_row - energy_np[energy_idx_pytorch], Vhs_row)[::-1]

        # V_next 计算
        # JAX: V_next = jnp.maximum(jnp.power(gamma, ii) * last_value + V_total[-1, :] - energy[-ii-1, :], last_value_reach)
        V_next_jax = np.maximum(
            (gamma ** ii) * last_value_np + V_total_jax[-1] - energy_np[energy_idx_jax],
            last_value_reach_np
        )
        # PyTorch: V_next = torch.maximum((gamma ** ii) * last_value + V_total[-1] - energy[energy_idx], last_value_reach)
        V_next_pytorch = np.maximum(
            (gamma ** ii) * last_value_np + V_total_pytorch[-1] - energy_np[energy_idx_pytorch],
            last_value_reach_np
        )

        # V_total_1
        V_total_1_jax = np.concatenate([V_total_jax, V_next_jax.reshape(1, -1)], axis=0)
        V_total_1_pytorch = np.concatenate([V_total_pytorch, V_next_pytorch.reshape(1, -1)], axis=0)

        # argmin
        index_1_jax = np.argmin(V_total_1_jax, axis=0)
        index_1_pytorch = np.argmin(V_total_1_pytorch, axis=0)

        trace.append({
            'ii': ii,
            'energy_idx_jax': energy_idx_jax,
            'energy_idx_pytorch': energy_idx_pytorch,
            'energy_jax': energy_np[energy_idx_jax].copy(),
            'energy_pytorch': energy_np[energy_idx_pytorch].copy(),
            'V_total_jax': V_total_jax.copy(),
            'V_total_pytorch': V_total_pytorch.copy(),
            'V_next_jax': V_next_jax.copy(),
            'V_next_pytorch': V_next_pytorch.copy(),
            'V_total_1_jax': V_total_1_jax.copy(),
            'V_total_1_pytorch': V_total_1_pytorch.copy(),
            'index_1_jax': index_1_jax.copy(),
            'index_1_pytorch': index_1_pytorch.copy(),
            'match': np.array_equal(index_1_jax, index_1_pytorch),
        })

        # 更新 done（使用 JAX 的索引）
        for j in range(nh):
            done[index_1_jax[j], j] = 1.0

        # 更新 mask_1
        mask_1 = np.roll(mask_1, 1, axis=0)
        mask_1[0] = 1.0

    return trace, done


def main():
    print("=" * 70)
    print("calculate_indexs3 对拍测试")
    print("=" * 70)

    # 小规模测试数据
    T = 5
    nh = 2
    gamma = 0.99

    # 固定随机种子
    torch.manual_seed(42)

    # 创建测试数据
    reward = torch.randn(T, nh).abs() * 100  # energy_consumption [T, nh]
    energy = torch.randn(T + 1, nh) * 200  # energy [T+1, nh]
    T_hs = torch.randn(T + 1, nh).abs() * 500  # reach (g_values) [T+1, nh]
    last_value = torch.randn(nh) * 100
    last_value_reach = torch.randn(nh).abs() * 500

    print(f"\n测试数据:")
    print(f"  T={T}, nh={nh}, gamma={gamma}")
    print(f"  reward (energy_consumption) shape: {reward.shape}")
    print(f"  energy shape: {energy.shape}")
    print(f"  T_hs (reach) shape: {T_hs.shape}")

    print(f"\n具体数值:")
    print(f"  reward:\n{reward}")
    print(f"  energy:\n{energy}")
    print(f"  T_hs:\n{T_hs}")
    print(f"  last_value: {last_value}")
    print(f"  last_value_reach: {last_value_reach}")

    # 运行 PyTorch 版本
    print(f"\n{'='*70}")
    print("运行 PyTorch 版本...")
    indexs_pt, done_pt, trace_pt = calculate_indexs3_pytorch(
        gamma, reward, energy, T_hs, last_value, last_value_reach
    )

    # 运行 JAX 手动模拟版本
    print("运行 JAX 手动模拟版本...")
    trace_jax, done_jax = calculate_indexs3_jax_manual(
        gamma, reward, energy, T_hs, last_value, last_value_reach
    )

    # 逐轮对比
    print(f"\n{'='*70}")
    print("逐轮对比结果:")
    print(f"{'='*70}")

    all_match = True
    energy_idx_mismatch_count = 0

    for i, (t_pt, t_jax) in enumerate(zip(trace_pt, trace_jax)):
        ii = t_pt['ii']
        print(f"\n--- Iteration {i}, ii={ii} ---")

        # 检查 energy 索引差异
        print(f"  energy_idx (PyTorch): {t_jax['energy_idx_pytorch']}")
        print(f"  energy_idx (JAX):     {t_jax['energy_idx_jax']}")

        if t_jax['energy_idx_jax'] != t_jax['energy_idx_pytorch']:
            energy_idx_mismatch_count += 1
            print(f"  ⚠️  ENERGY INDEX DIFFERS!")
            print(f"     JAX uses energy[{t_jax['energy_idx_jax']}] = {t_jax['energy_jax']}")
            print(f"     PyTorch uses energy[{t_jax['energy_idx_pytorch']}] = {t_jax['energy_pytorch']}")

        # 检查 V_total 差异
        v_total_diff = np.abs(t_jax['V_total_jax'] - t_jax['V_total_pytorch']).max()
        print(f"  V_total max diff: {v_total_diff:.6f}")
        if v_total_diff > 1e-5:
            print(f"  ⚠️  V_total differs!")
            print(f"     JAX V_total: {t_jax['V_total_jax'].flatten()}")
            print(f"     PT  V_total: {t_jax['V_total_pytorch'].flatten()}")

        # 检查 V_next 差异
        v_next_diff = np.abs(t_jax['V_next_jax'] - t_jax['V_next_pytorch']).max()
        print(f"  V_next max diff: {v_next_diff:.6f}")

        # 检查 index_1 差异
        match = t_jax['match']
        print(f"  index_1 match: {match}")
        print(f"    JAX: {t_jax['index_1_jax']}")
        print(f"    PT:  {t_jax['index_1_pytorch']}")

        if not match:
            all_match = False
            print(f"  ❌ INDEX MISMATCH!")

    # 最终对比
    print(f"\n{'='*70}")
    print("最终结果对比:")
    print(f"{'='*70}")

    print(f"\nindexs (PyTorch):\n{indexs_pt}")
    print(f"\ndone (PyTorch):\n{done_pt}")
    print(f"\ndone (JAX manual):\n{done_jax}")

    done_diff = np.abs(done_pt.numpy() - done_jax).max()
    print(f"\ndone max diff: {done_diff:.6f}")

    # 总结
    print(f"\n{'='*70}")
    print("总结:")
    print(f"{'='*70}")

    print(f"\n1. Energy 索引差异:")
    print(f"   迭代次数: {len(trace_pt)}")
    print(f"   energy_idx 不同的次数: {energy_idx_mismatch_count}")
    if energy_idx_mismatch_count > 0:
        print(f"   ⚠️  PyTorch 和 JAX 使用了不同的 energy 索引！")
        print(f"   但是：由于 energy 是 [T+1, nh] 形状，")
        print(f"   JAX 访问 energy[0:T]（即 energy[-T:] 到 energy[-1:]），")
        print(f"   PyTorch 访问 energy[0:T-1]（即 energy[0] 到 energy[T-2]）。")
        print(f"   差异在于 JAX 使用了 energy[T]（最后一个元素），而 PyTorch 没有。")

    print(f"\n2. index_1 匹配情况:")
    if all_match:
        print(f"   ✅ 所有迭代的 index_1 都匹配")
    else:
        print(f"   ❌ 存在不匹配的迭代")

    print(f"\n3. done 矩阵差异:")
    if done_diff < 1e-6:
        print(f"   ✅ done 矩阵完全一致 (max diff = {done_diff:.6f})")
    else:
        print(f"   ❌ done 矩阵存在差异 (max diff = {done_diff:.6f})")

    print(f"\n4. 结论:")
    if energy_idx_mismatch_count > 0 and not all_match:
        print(f"   ❌ 问题A确认：energy 索引存在差异，且导致 index_1 不同")
        print(f"   修复建议：将 PyTorch 的 energy_idx = T - ii - 1 改为 energy_idx = T - ii")
    elif energy_idx_mismatch_count > 0 and all_match:
        print(f"   ⚠️  energy 索引有差异，但 index_1 仍匹配（可能是因为 energy 值恰好不影响 argmin）")
        print(f"   建议：修复 energy 索引以确保完全一致")
    else:
        print(f"   ✅ 问题A不成立：energy 索引完全一致")


if __name__ == "__main__":
    main()
