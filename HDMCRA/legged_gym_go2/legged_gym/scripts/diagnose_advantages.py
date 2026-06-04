#!/usr/bin/env python3
"""
EC-EFPPO 诊断脚本：分析三路优势量级分布和 earliest reach index 信号质量。

运行方式：
  python legged_gym_go2/legged_gym/scripts/diagnose_advantages.py \
    --checkpoint-path logs/ecfppo_go2/20260604-105543/model_100.pt \
    --num-steps 50 --num-envs 64
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
    parser = argparse.ArgumentParser(description="EC-EFPPO 优势信号诊断")
    parser.add_argument("--checkpoint-path", required=True, help="EC-EFPPO checkpoint path")
    parser.add_argument("--num-steps", type=int, default=50, help="Rollout steps")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel envs")
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

    # 解析 low-level model 路径
    if args.low_level_model:
        train_cfg.runner.low_level_model_path = args.low_level_model
    elif "low_level_model_path" in checkpoint:
        train_cfg.runner.low_level_model_path = checkpoint["low_level_model_path"]

    env = create_env(env_cfg, train_cfg, sim_args, device)

    # 从 checkpoint 推断网络结构
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

    # 收集 rollout 数据
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

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps} | "
                  f"g_mean={g_vals.mean():.3f} | h_mean={h_vals.mean():.3f} | "
                  f"energy_mean={energy.mean():.1f}")

    return buffer


def diagnose_earliest_reach_index(buffer, gamma_energy):
    """诊断 earliest reach index 信号质量。"""
    print(f"\n{'='*60}")
    print("Earliest Reach Index 诊断")
    print(f"{'='*60}")

    # 准备数据
    reach_append = buffer.g_values  # [T+1, N]
    energy_append = buffer.energy  # [T+1, N]
    energy_consumption = buffer.energy_consumption  # [T, N]

    # 获取 bootstrap values
    last_values_energy = buffer.values[-1]  # [N]
    last_values_reach = buffer.value_reach[-1]  # [N]

    print(f"\n输入数据统计:")
    print(f"  reach (g_values) shape: {reach_append.shape}")
    print(f"  reach 范围: [{reach_append.min():.3f}, {reach_append.max():.3f}], mean={reach_append.mean():.3f}")
    print(f"  energy shape: {energy_append.shape}")
    print(f"  energy 范围: [{energy_append.min():.1f}, {energy_append.max():.1f}], mean={energy_append.mean():.1f}")
    print(f"  energy_consumption shape: {energy_consumption.shape}")
    print(f"  energy_consumption 范围: [{energy_consumption.min():.3f}, {energy_consumption.max():.3f}], mean={energy_consumption.mean():.3f}")

    # 调用 calculate_indexs3
    print(f"\n调用 calculate_indexs3(gamma_energy={gamma_energy})...")
    Vs, done = calculate_indexs3(
        gamma_energy,
        energy_consumption,
        energy_append,
        reach_append,
        last_values_energy,
        last_values_reach,
    )

    print(f"\ncalculate_indexs3 输出:")
    print(f"  Vs shape: {Vs.shape}, dtype: {Vs.dtype}")
    Vs_float = Vs.float()
    print(f"  Vs 范围: [{Vs_float.min():.3f}, {Vs_float.max():.3f}], mean={Vs_float.mean():.3f}")
    print(f"  done shape: {done.shape}, dtype: {done.dtype}")
    print(f"  done 范围: [{done.min():.3f}, {done.max():.3f}], mean={done.float().mean():.3f}")

    # 分析 done 矩阵
    done_float = done.float()
    done_binary = (done_float > 0.5).float()
    done_per_step = done_binary.sum(dim=1)  # [T+1]
    done_per_env = done_binary.sum(dim=0)  # [N]

    print(f"\ndone 矩阵分析:")
    print(f"  每步触发 done 的环境数: min={done_per_step.min():.0f}, max={done_per_step.max():.0f}, mean={done_per_step.mean():.1f}")
    print(f"  每环境触发 done 的步数: min={done_per_env.min():.0f}, max={done_per_env.max():.0f}, mean={done_per_env.mean():.1f}")

    # 检查是否每个环境最多只有一个 done
    multi_done_envs = (done_per_env > 1).sum().item()
    no_done_envs = (done_per_env == 0).sum().item()
    single_done_envs = (done_per_env == 1).sum().item()

    print(f"\ndone 分布:")
    print(f"  无 done 的环境: {no_done_envs} ({no_done_envs/buffer.num_envs*100:.1f}%)")
    print(f"  单个 done 的环境: {single_done_envs} ({single_done_envs/buffer.num_envs*100:.1f}%)")
    print(f"  多个 done 的环境: {multi_done_envs} ({multi_done_envs/buffer.num_envs*100:.1f}%)")

    if multi_done_envs > 0:
        print(f"  ⚠️ 警告: {multi_done_envs} 个环境有多个 done 标志，这可能表示 earliest reach index 计算有问题")

    # 找到每个环境的第一个 done 位置
    first_done = torch.argmax(done_binary, dim=0)  # [N]
    envs_with_done = done_per_env > 0

    if envs_with_done.any():
        first_done_valid = first_done[envs_with_done]
        print(f"\n第一个 done 位置统计 (有 done 的环境):")
        print(f"  范围: [{first_done_valid.min().item()}, {first_done_valid.max().item()}]")
        print(f"  均值: {first_done_valid.float().mean().item():.1f}")

        # 检查 done 位置对应的 reach 值
        done_reach_values = []
        for env_idx in range(buffer.num_envs):
            if envs_with_done[env_idx]:
                done_step = first_done[env_idx].item()
                if done_step < reach_append.shape[0]:
                    done_reach_values.append(reach_append[done_step, env_idx].item())

        if done_reach_values:
            done_reach_tensor = torch.tensor(done_reach_values)
            print(f"\ndone 位置对应的 reach (g) 值:")
            print(f"  范围: [{done_reach_tensor.min():.3f}, {done_reach_tensor.max():.3f}]")
            print(f"  均值: {done_reach_tensor.mean():.3f}")
            print(f"  负值比例: {(done_reach_tensor < 0).float().mean()*100:.1f}%")

    return done, Vs


def diagnose_advantages(buffer, gamma_energy, gamma_reach, gae_lambda, gamma_reach_init, done):
    """诊断三路优势的量级分布。"""
    print(f"\n{'='*60}")
    print("三路优势量级分布诊断")
    print(f"{'='*60}")

    # 准备数据
    reach_append = buffer.g_values  # [T+1, N]
    energy_append = buffer.energy  # [T+1, N]
    energy_consumption = buffer.energy_consumption  # [T, N]

    # 构造 value 序列
    V_reach_append = torch.cat(
        [buffer.value_reach, buffer.value_reach[-1:].clone()], dim=0
    )
    V_energy_append = torch.cat(
        [buffer.values, buffer.values[-1:].clone()], dim=0
    )

    # 组合信号
    V_total_append = torch.maximum(V_reach_append, V_energy_append - energy_append)
    g_append = torch.maximum(reach_append, -energy_append)

    # done_for_gae
    done_for_gae = done[:-1, :].clone()  # [T, N]
    done_for_gae = (done_for_gae.bool() | buffer.dones).float()

    print(f"\n输入信号统计:")
    print(f"  reach_append (g) 范围: [{reach_append.min():.3f}, {reach_append.max():.3f}], mean={reach_append.mean():.3f}")
    print(f"  energy_append 范围: [{energy_append.min():.1f}, {energy_append.max():.1f}], mean={energy_append.mean():.1f}")
    print(f"  V_reach_append 范围: [{V_reach_append.min():.3f}, {V_reach_append.max():.3f}], mean={V_reach_append.mean():.3f}")
    print(f"  V_energy_append 范围: [{V_energy_append.min():.3f}, {V_energy_append.max():.3f}], mean={V_energy_append.mean():.3f}")
    print(f"  g_append = max(reach, -energy) 范围: [{g_append.min():.3f}, {g_append.max():.3f}], mean={g_append.mean():.3f}")
    print(f"  V_total = max(V_reach, V_energy-energy) 范围: [{V_total_append.min():.3f}, {V_total_append.max():.3f}], mean={V_total_append.mean():.3f}")
    print(f"  done_for_gae 范围: [{done_for_gae.min():.3f}, {done_for_gae.max():.3f}], mean={done_for_gae.mean():.3f}")

    # 计算三路优势
    print(f"\n计算三路优势...")

    # 1. Reach advantage
    advantages_h, targets_h = calculate_reach_gae(
        gamma_reach, gae_lambda, reach_append, V_reach_append, done_for_gae
    )

    # 2. Energy advantage
    advantages_V, targets_V = calculate_energy_gae(
        gamma_energy, gae_lambda, energy_consumption, buffer.values,
        done_for_gae, buffer.values[-1]
    )

    # 3. Combined advantage
    advantages_total, _ = calculate_reach_gae(
        gamma_reach_init, gae_lambda, g_append, V_total_append, done_for_gae
    )

    # 统计量
    print(f"\n三路优势统计:")
    print(f"  {'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Abs Mean':>10}")
    print(f"  {'-'*75}")

    for name, adv in [("Reach Advantage", advantages_h),
                       ("Energy Advantage", advantages_V),
                       ("Combined Advantage", advantages_total)]:
        print(f"  {name:<25} {adv.min():>10.3f} {adv.max():>10.3f} {adv.mean():>10.3f} {adv.std():>10.3f} {adv.abs().mean():>10.3f}")

    # 计算量级比
    reach_abs_mean = advantages_h.abs().mean().item()
    energy_abs_mean = advantages_V.abs().mean().item()
    combined_abs_mean = advantages_total.abs().mean().item()

    print(f"\n量级比:")
    print(f"  |Reach| / |Energy| = {reach_abs_mean / (energy_abs_mean + 1e-8):.2f}")
    print(f"  |Combined| / |Reach| = {combined_abs_mean / (reach_abs_mean + 1e-8):.2f}")
    print(f"  |Combined| / |Energy| = {combined_abs_mean / (energy_abs_mean + 1e-8):.2f}")

    # 分析 combined advantage 的构成
    print(f"\nCombined Advantage 构成分析:")
    print(f"  g_append = max(reach, -energy)")
    print(f"  V_total = max(V_reach, V_energy - energy)")

    # 检查 max 操作的选择比例
    reach_dominant = (reach_append > -energy_append).float().mean().item()
    energy_dominant = (1 - reach_dominant)
    print(f"  g_append 中 reach 主导的比例: {reach_dominant*100:.1f}%")
    print(f"  g_append 中 -energy 主导的比例: {energy_dominant*100:.1f}%")

    V_reach_dominant = (V_reach_append > V_energy_append - energy_append).float().mean().item()
    V_energy_dominant = (1 - V_reach_dominant)
    print(f"  V_total 中 V_reach 主导的比例: {V_reach_dominant*100:.1f}%")
    print(f"  V_total 中 (V_energy-energy) 主导的比例: {V_energy_dominant*100:.1f}%")

    # targets 统计
    print(f"\nTargets 统计:")
    print(f"  {'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*65}")
    for name, tgt in [("Targets Energy", targets_V),
                       ("Targets Reach", targets_h)]:
        print(f"  {name:<25} {tgt.min():>10.3f} {tgt.max():>10.3f} {tgt.mean():>10.3f} {tgt.std():>10.3f}")

    return advantages_h, advantages_V, advantages_total


def main():
    args = parse_args()
    env, actor_critic, device, train_cfg, checkpoint = build_env_and_policy(args)

    try:
        # 收集 rollout 数据
        buffer = collect_rollout_data(env, actor_critic, device, args.num_steps)

        # 获取超参数
        gamma_energy = train_cfg.algorithm.gamma_energy
        gamma_reach = train_cfg.algorithm.gamma_reach_init  # 使用初始值
        gae_lambda = train_cfg.algorithm.gae_lambda
        gamma_reach_init = train_cfg.algorithm.gamma_reach_init

        print(f"\n超参数:")
        print(f"  gamma_energy: {gamma_energy}")
        print(f"  gamma_reach: {gamma_reach}")
        print(f"  gae_lambda: {gae_lambda}")
        print(f"  gamma_reach_init: {gamma_reach_init}")

        # 诊断 earliest reach index
        done, Vs = diagnose_earliest_reach_index(buffer, gamma_energy)

        # 诊断三路优势
        adv_h, adv_V, adv_total = diagnose_advantages(
            buffer, gamma_energy, gamma_reach, gae_lambda, gamma_reach_init, done
        )

        # 最终诊断结论
        print(f"\n{'='*60}")
        print("诊断结论")
        print(f"{'='*60}")

        reach_magnitude = adv_h.abs().mean().item()
        energy_magnitude = adv_V.abs().mean().item()
        combined_magnitude = adv_total.abs().mean().item()

        if reach_magnitude > 10 * energy_magnitude:
            print(f"⚠️  Reach 优势量级 ({reach_magnitude:.3f}) 远大于 Energy 优势 ({energy_magnitude:.3f})")
            print(f"   比值: {reach_magnitude/energy_magnitude:.1f}x")
            print(f"   Combined advantage 主要被 Reach 信号驱动")
        elif energy_magnitude > 10 * reach_magnitude:
            print(f"⚠️  Energy 优势量级 ({energy_magnitude:.3f}) 远大于 Reach 优势 ({reach_magnitude:.3f})")
            print(f"   比值: {energy_magnitude/reach_magnitude:.1f}x")
            print(f"   Combined advantage 主要被 Energy 信号驱动")
        else:
            print(f"✓  Reach 和 Energy 优势量级相对平衡")
            print(f"   Reach: {reach_magnitude:.3f}, Energy: {energy_magnitude:.3f}")

        if combined_magnitude > 100:
            print(f"⚠️  Combined advantage 量级过大 ({combined_magnitude:.3f})，可能导致梯度不稳定")

    finally:
        env.close()


if __name__ == "__main__":
    main()
