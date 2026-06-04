#!/usr/bin/env python3
"""
EC-EFPPO 三路优势信号详细诊断脚本。

深入分析 combined advantage 的计算过程，捕获所有中间变量，
并与 JAX 参考实现的行为进行对比。

运行方式：
  python legged_gym_go2/legged_gym/scripts/diagnose_advantage_detail.py \
    --checkpoint-path logs/ecfppo_go2/20260604-105543/model_100.pt \
    --num-steps 20 --num-envs 8
"""

import argparse
import os
import sys
from datetime import datetime

import isaacgym
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2EC_EFPPOCfgPPO
from legged_gym.scripts.train_ecfppo import HierarchicalVecEnv, create_env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args

from rsl_rl.algorithms.ecfppo import EC_EFPPO_Buffer
from rsl_rl.algorithms.ecfppo_gae import calculate_indexs3, calculate_reach_gae, calculate_energy_gae
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EC-EFPPO 三路优势详细诊断")
    parser.add_argument("--checkpoint-path", required=True, help="EC-EFPPO checkpoint path")
    parser.add_argument("--num-steps", type=int, default=20, help="Rollout steps")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--low-level-model", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def build_env_and_policy(args):
    """创建环境和加载策略。"""
    sim_args = get_args()
    sim_args.headless = not args.render
    sim_args.num_envs = args.num_envs
    device = torch.device(sim_args.rl_device)

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2EC_EFPPOCfgPPO()
    env_cfg.env.num_envs = args.num_envs
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, sim_args)

    if args.low_level_model:
        train_cfg.runner.low_level_model_path = args.low_level_model
    elif "low_level_model_path" in checkpoint:
        train_cfg.runner.low_level_model_path = checkpoint["low_level_model_path"]

    env = create_env(env_cfg, train_cfg, sim_args, device)

    state_dict = checkpoint.get("actor_critic", {})
    hidden_dim = None
    num_hidden_layers = None
    for key in state_dict:
        if key.startswith("actor.") and key.endswith(".weight"):
            hidden_dim = state_dict[key].shape[0]
            linear_count = sum(1 for k in state_dict if k.startswith("actor.") and k.endswith(".weight"))
            num_hidden_layers = linear_count - 1
            break

    actor_critic = EC_EFPPO_ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        hidden_dim=hidden_dim or train_cfg.network.hidden_dim,
        num_hidden_layers=num_hidden_layers or train_cfg.network.num_hidden_layers,
        init_noise_std=1.0,
        activation=train_cfg.network.activation,
    ).to(device)

    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()

    return env, actor_critic, device, train_cfg, checkpoint


def collect_rollout_data(env, actor_critic, device, num_steps):
    """收集一个 rollout 的完整数据。"""
    buffer = EC_EFPPO_Buffer(
        num_envs=env.num_envs,
        horizon=num_steps,
        obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
        device=device,
    )

    obs, g_vals, h_vals, energy = env.reset()
    obs = obs.to(device)

    print(f"\n{'='*60}")
    print(f"开始收集 rollout 数据 ({num_steps} steps, {env.num_envs} envs)")
    print(f"{'='*60}")

    for step in range(num_steps):
        with torch.no_grad():
            actions, log_probs, energy_value, reach_value = actor_critic.act(obs)

        obs_new, g_vals_new, h_vals_new, dones, infos, energy_new, energy_consumption = env.step(actions)

        buffer.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=energy_value,
            value_reach=reach_value,
            energy=energy,
            energy_consumption=energy_consumption,
            g_values=g_vals,
            h_values=h_vals,
            dones=dones,
            next_obs=obs_new.to(device),
            next_energy=energy_new,
            next_g=g_vals_new,
            next_h=h_vals_new,
        )

        obs = obs_new.to(device)
        g_vals = g_vals_new
        h_vals = h_vals_new
        energy = energy_new

    return buffer, obs


def diagnose_advantage_computation(buffer, last_values_energy, last_values_reach,
                                    gamma_energy, gamma_reach, gae_lambda, gamma_reach_init):
    """详细诊断优势计算过程。"""

    print(f"\n{'='*60}")
    print("三路优势计算详细诊断")
    print(f"{'='*60}")

    T = buffer.horizon
    N = buffer.num_envs

    # Step 1: 构造扩展序列
    print(f"\n--- Step 1: 构造扩展序列 ---")

    reach_append = buffer.g_values  # [T+1, N]
    V_reach_append = torch.cat([buffer.value_reach, last_values_reach.unsqueeze(0)], dim=0)
    energy_append = buffer.energy  # [T+1, N]
    V_energy_append = torch.cat([buffer.values, last_values_energy.unsqueeze(0)], dim=0)

    print(f"reach_append (g_values) shape: {reach_append.shape}")
    print(f"  范围: [{reach_append.min():.3f}, {reach_append.max():.3f}], mean={reach_append.mean():.3f}")

    print(f"V_reach_append shape: {V_reach_append.shape}")
    print(f"  范围: [{V_reach_append.min():.3f}, {V_reach_append.max():.3f}], mean={V_reach_append.mean():.3f}")

    print(f"energy_append shape: {energy_append.shape}")
    print(f"  范围: [{energy_append.min():.1f}, {energy_append.max():.1f}], mean={energy_append.mean():.1f}")

    print(f"V_energy_append shape: {V_energy_append.shape}")
    print(f"  范围: [{V_energy_append.min():.3f}, {V_energy_append.max():.3f}], mean={V_energy_append.mean():.3f}")

    # Step 2: 组合信号
    print(f"\n--- Step 2: 组合信号 ---")

    # g_append = max(reach, -energy)
    g_append = torch.maximum(reach_append, -energy_append)
    print(f"g_append = max(reach, -energy)")
    print(f"  reach 主导比例: {(reach_append > -energy_append).float().mean()*100:.1f}%")
    print(f"  -energy 主导比例: {(reach_append <= -energy_append).float().mean()*100:.1f}%")
    print(f"  范围: [{g_append.min():.3f}, {g_append.max():.3f}], mean={g_append.mean():.3f}")

    # V_total = max(V_reach, V_energy - energy)
    V_total_append = torch.maximum(V_reach_append, V_energy_append - energy_append)
    print(f"\nV_total = max(V_reach, V_energy - energy)")
    print(f"  V_reach 主导比例: {(V_reach_append > V_energy_append - energy_append).float().mean()*100:.1f}%")
    print(f"  (V_energy-energy) 主导比例: {(V_reach_append <= V_energy_append - energy_append).float().mean()*100:.1f}%")
    print(f"  范围: [{V_total_append.min():.3f}, {V_total_append.max():.3f}], mean={V_total_append.mean():.3f}")

    # 分析 -energy 的分布
    neg_energy = -energy_append
    print(f"\n-energy 分布:")
    print(f"  范围: [{neg_energy.min():.1f}, {neg_energy.max():.1f}], mean={neg_energy.mean():.1f}")
    print(f"  负值比例 (energy>0): {(neg_energy < 0).float().mean()*100:.1f}%")
    print(f"  正值比例 (energy<0): {(neg_energy >= 0).float().mean()*100:.1f}%")

    # Step 3: 计算 done 矩阵
    print(f"\n--- Step 3: 计算 done 矩阵 ---")

    _, done = calculate_indexs3(
        gamma_energy,
        buffer.energy_consumption,
        energy_append,
        reach_append,
        last_values_energy,
        last_values_reach,
    )

    done_for_gae = done[:-1, :].clone()
    done_for_gae = (done_for_gae.bool() | buffer.dones).float()

    print(f"done shape: {done.shape}")
    print(f"done_for_gae shape: {done_for_gae.shape}")
    print(f"done_for_gae mean: {done_for_gae.mean():.3f}")
    print(f"每个环境的 done 数量: min={done_for_gae.sum(dim=0).min():.0f}, max={done_for_gae.sum(dim=0).max():.0f}")

    # Step 4: 计算三路优势
    print(f"\n--- Step 4: 计算三路优势 ---")

    # Reach advantage
    advantages_h, targets_h = calculate_reach_gae(
        gamma_reach, gae_lambda, reach_append, V_reach_append, done_for_gae
    )

    # Energy advantage
    advantages_V, targets_V = calculate_energy_gae(
        gamma_energy, gae_lambda, buffer.energy_consumption, buffer.values,
        done_for_gae, last_values_energy
    )

    # Combined advantage
    advantages_total, _ = calculate_reach_gae(
        gamma_reach_init, gae_lambda, g_append, V_total_append, done_for_gae
    )

    print(f"\n三路优势统计:")
    print(f"  {'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'AbsMean':>10}")
    print(f"  {'-'*75}")

    for name, adv in [("Reach Advantage", advantages_h),
                       ("Energy Advantage", advantages_V),
                       ("Combined Advantage", advantages_total)]:
        print(f"  {name:<25} {adv.min():>10.3f} {adv.max():>10.3f} {adv.mean():>10.3f} {adv.std():>10.3f} {adv.abs().mean():>10.3f}")

    # Step 5: 分析 combined advantage 的构成
    print(f"\n--- Step 5: Combined Advantage 构成分析 ---")

    # 归一化后的 advantage
    adv_norm = (advantages_total - advantages_total.mean()) / (advantages_total.std() + 1e-8)
    print(f"归一化后的 Combined Advantage:")
    print(f"  范围: [{adv_norm.min():.3f}, {adv_norm.max():.3f}], mean={adv_norm.mean():.3f}, std={adv_norm.std():.3f}")

    # 分析 advantage 的符号分布
    pos_ratio = (advantages_total > 0).float().mean()
    neg_ratio = (advantages_total < 0).float().mean()
    print(f"\nCombined Advantage 符号分布:")
    print(f"  正值比例: {pos_ratio*100:.1f}%")
    print(f"  负值比例: {neg_ratio*100:.1f}%")

    # Step 6: 分析 targets
    print(f"\n--- Step 6: Targets 统计 ---")
    print(f"  {'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*65}")
    for name, tgt in [("Targets Energy", targets_V),
                       ("Targets Reach", targets_h)]:
        print(f"  {name:<25} {tgt.min():>10.3f} {tgt.max():>10.3f} {tgt.mean():>10.3f} {tgt.std():>10.3f}")

    # Step 7: 诊断结论
    print(f"\n--- Step 7: 诊断结论 ---")

    reach_magnitude = advantages_h.abs().mean().item()
    energy_magnitude = advantages_V.abs().mean().item()
    combined_magnitude = advantages_total.abs().mean().item()

    print(f"\n优势量级:")
    print(f"  |Reach| = {reach_magnitude:.3f}")
    print(f"  |Energy| = {energy_magnitude:.3f}")
    print(f"  |Combined| = {combined_magnitude:.3f}")
    print(f"  |Reach|/|Energy| = {reach_magnitude/(energy_magnitude+1e-8):.1f}x")

    # 分析 g_append 的构成
    print(f"\ng_append 构成:")
    print(f"  reach 主导时的 g 均值: {g_append[reach_append > -energy_append].mean():.3f}")
    print(f"  -energy 主导时的 g 均值: {g_append[reach_append <= -energy_append].mean():.3f}")

    # 分析能量状态
    print(f"\n能量状态分析:")
    print(f"  energy < 0 的比例: {(energy_append < 0).float().mean()*100:.1f}%")
    print(f"  energy = -400 的比例: {(energy_append == -400).float().mean()*100:.1f}%")
    print(f"  energy > 0 的比例: {(energy_append > 0).float().mean()*100:.1f}%")

    return {
        'reach_append': reach_append,
        'energy_append': energy_append,
        'g_append': g_append,
        'V_total_append': V_total_append,
        'done_for_gae': done_for_gae,
        'advantages_h': advantages_h,
        'advantages_V': advantages_V,
        'advantages_total': advantages_total,
        'targets_h': targets_h,
        'targets_V': targets_V,
    }


def main():
    args = parse_args()
    env, actor_critic, device, train_cfg, checkpoint = build_env_and_policy(args)

    try:
        # 收集 rollout 数据
        buffer, obs = collect_rollout_data(env, actor_critic, device, args.num_steps)

        # 获取 bootstrap values
        with torch.no_grad():
            last_values_energy, last_values_reach = actor_critic.evaluate(obs)

        # 获取超参数
        gamma_energy = train_cfg.algorithm.gamma_energy
        gamma_reach = train_cfg.algorithm.gamma_reach_init
        gae_lambda = train_cfg.algorithm.gae_lambda
        gamma_reach_init = train_cfg.algorithm.gamma_reach_init

        print(f"\n超参数:")
        print(f"  gamma_energy: {gamma_energy}")
        print(f"  gamma_reach: {gamma_reach}")
        print(f"  gae_lambda: {gae_lambda}")
        print(f"  gamma_reach_init: {gamma_reach_init}")

        # 运行诊断
        results = diagnose_advantage_computation(
            buffer, last_values_energy, last_values_reach,
            gamma_energy, gamma_reach, gae_lambda, gamma_reach_init
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
