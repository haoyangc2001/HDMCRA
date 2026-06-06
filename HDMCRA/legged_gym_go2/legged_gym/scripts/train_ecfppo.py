#!/usr/bin/env python3
"""
EC-EFPPO 训练脚本。

基于 train_reach_avoid.py 改造，适配 EC-EFPPO 三网络架构。
核心改动：
- 使用 EC_EFPPO_ActorCritic（三网络）替代 ActorCritic（单网络）
- 使用 EC_EFPPO 替代 ReachAvoidPPO（三个独立优化器）
- 收集 energy/energy_consumption 数据到 buffer
- 三路优势计算 + 三路独立 PPO 更新
- γ_reach 和 entropy 退火
- 日志增加 EC-EFPPO 特有指标
"""

import os
import time
from datetime import datetime
from typing import Tuple

import isaacgym
import torch

from legged_gym.envs.go2.hierarchical_go2_env import HierarchicalGO2Env
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2EC_EFPPOCfgPPO
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args

from rsl_rl.algorithms.ecfppo import EC_EFPPO
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


class HierarchicalVecEnv:
    """封装 HierarchicalGO2Env，提供向量化接口。"""

    def __init__(self, env: HierarchicalGO2Env):
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions
        self.device = env.device

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (obs, g_vals, h_vals, energy)。"""
        obs, g_vals, h_vals, energy = self.env.reset()
        return obs, g_vals, h_vals, energy

    def step(self, actions: torch.Tensor):
        """返回 (obs, g_vals, h_vals, dones, infos, energy, energy_consumption)。"""
        obs, g_vals, h_vals, dones, infos, energy, energy_consumption = self.env.step(actions)
        return obs, g_vals, h_vals, dones, infos, energy, energy_consumption

    def close(self) -> None:
        self.env.close()


def create_env(env_cfg, train_cfg, args, device) -> HierarchicalVecEnv:
    """创建环境实例。"""
    base_env = HierarchicalGO2Env(
        cfg=env_cfg,
        low_level_model_path=train_cfg.runner.low_level_model_path,
        args=args,
        device=device,
    )
    return HierarchicalVecEnv(base_env)


def compute_reach_avoid_success_rate(
    g_sequence: torch.Tensor,
    h_sequence: torch.Tensor,
    energy_sequence: torch.Tensor = None,
):
    """
    计算 Reach-Avoid 任务的成功率、执行成本和能量消耗。

    在原有基础上增加能量消耗统计。

    Args:
        g_sequence: [T, N] reach 值序列
        h_sequence: [T, N] 安全约束值序列
        energy_sequence: [T+1, N] 能量序列（可选），用于计算成功环境的平均能量消耗

    Returns:
        success_rate: 成功率
        execution_cost: 成功环境的平均到达时间步
        avg_energy_consumption: 成功环境的平均能量消耗（如果提供 energy_sequence）
    """
    with torch.no_grad():
        time_steps, num_envs = g_sequence.shape

        g_negative = g_sequence < 0
        has_success = g_negative.any(dim=0)
        first_success = torch.argmax(g_negative.long(), dim=0)
        first_indices = torch.where(
            has_success,
            first_success,
            torch.full((num_envs,), time_steps, device=g_sequence.device, dtype=torch.long),
        )

        time_index = torch.arange(time_steps, device=g_sequence.device).unsqueeze(1)
        before_success = time_index < first_indices.unsqueeze(0)
        h_violation = (h_sequence >= 0) & before_success
        safe_before = ~h_violation.any(dim=0)
        success = has_success & safe_before

        success_rate = success.float().mean().item()

        success_first_indices = first_indices[success]
        if success.sum().item() > 0:
            execution_cost = success_first_indices.float().mean().item()
        else:
            execution_cost = 0.0

        # 能量消耗统计
        avg_energy_consumption = 0.0
        if energy_sequence is not None and success.sum().item() > 0:
            # g/h 序列长度为 T，对应 rollout 中每一步后的状态；energy 序列长度为 T+1，
            # 其中 energy[0] 是初始状态，energy[t+1] 才是执行第 t 步动作后的能量。
            # 因此第一次到达目标的 g 索引 first_indices 需要映射到 energy 的 first_indices + 1。
            init_energy = energy_sequence[0]  # [N]
            energy_indices = torch.where(
                has_success,
                first_indices + 1,
                torch.full_like(first_indices, time_steps),
            )
            reach_energy = energy_sequence[energy_indices, torch.arange(num_envs)]  # [N]
            energy_used = init_energy - reach_energy  # [N] 正值表示消耗
            avg_energy_consumption = energy_used[success].mean().item()

        return success_rate, execution_cost, avg_energy_consumption


def train_ecfppo(args) -> None:
    """EC-EFPPO 主训练函数。"""

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2EC_EFPPOCfgPPO()

    # Plan A: 从配置读取网络结构（对齐基线 4×512+elu）
    net_cfg = train_cfg.network

    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    device = torch.device(args.rl_device)
    env = create_env(env_cfg, train_cfg, args, device)

    num_envs = env.num_envs
    obs_shape = (env.num_obs,)
    action_shape = (env.num_actions,)
    horizon = train_cfg.algorithm.num_steps_per_env

    # ---- 初始化 EC_EFPPO_ActorCritic ----
    actor_critic = EC_EFPPO_ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        hidden_dim=net_cfg.hidden_dim,
        num_hidden_layers=net_cfg.num_hidden_layers,
        init_noise_std=getattr(train_cfg.algorithm, 'init_noise_std', 1.0),
        activation=net_cfg.activation,
        log_std_min=getattr(train_cfg.algorithm, 'log_std_min', -5.0),
        log_std_max=getattr(train_cfg.algorithm, 'log_std_max', 2.0),
    )

    # ---- 初始化 EC_EFPPO ----
    # 获取 energy critic 专用的梯度裁剪值（如果配置中有的话）
    max_grad_norm_energy = getattr(train_cfg.algorithm, 'max_grad_norm_energy', None)

    alg = EC_EFPPO(
        actor_critic=actor_critic,
        learning_rate=train_cfg.algorithm.learning_rate,
        gamma_energy=train_cfg.algorithm.gamma_energy,
        gamma_reach_init=train_cfg.algorithm.gamma_reach_init,
        gamma_reach_final=train_cfg.algorithm.gamma_reach_final,
        gae_lambda=train_cfg.algorithm.gae_lambda,
        num_learning_epochs=train_cfg.algorithm.num_learning_epochs,
        num_mini_batches=train_cfg.algorithm.num_mini_batches,
        clip_param=train_cfg.algorithm.clip_eps,
        value_loss_coef=train_cfg.algorithm.vf_coef,
        entropy_coef=train_cfg.algorithm.entropy_coef,
        max_grad_norm=train_cfg.algorithm.max_grad_norm,
        max_grad_norm_energy=max_grad_norm_energy,
        anneal_entropy=train_cfg.algorithm.anneal_entropy,
        device=str(device),
    )
    alg.init_storage(num_envs, horizon, obs_shape, action_shape)

    # ---- 恢复 checkpoint（如果指定）----
    if train_cfg.runner.resume and os.path.exists(train_cfg.runner.resume_path):
        print(f"Resuming from {train_cfg.runner.resume_path}")
        ckpt = torch.load(train_cfg.runner.resume_path, map_location=device)
        actor_critic.load_state_dict(ckpt["actor_critic"])
        # 注意：三个优化器的 state_dict 需要分别恢复。旧 checkpoint 可能缺少 std optimizer state。
        if "policy_optimizer" in ckpt:
            try:
                alg.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
            except ValueError as exc:
                print(f"Warning: skipped policy optimizer state due to parameter mismatch: {exc}")
        if "energy_optimizer" in ckpt:
            alg.energy_optimizer.load_state_dict(ckpt["energy_optimizer"])
        if "reach_optimizer" in ckpt:
            alg.reach_optimizer.load_state_dict(ckpt["reach_optimizer"])
        # 恢复归一化统计量
        if "obs_rms_state" in ckpt and hasattr(env.env, 'high_level_env'):
            env.env.high_level_env.set_obs_rms_state(ckpt["obs_rms_state"])
            print(f"Loaded observation normalization stats from checkpoint")
        if "energy_target_rms_state" in ckpt:
            alg.set_energy_target_rms_state(ckpt["energy_target_rms_state"])
            print(f"Loaded energy target normalization stats from checkpoint")
        start_iteration = ckpt.get("iteration", 0)
    else:
        start_iteration = 0

    # ---- 日志目录 ----
    log_dir = os.path.join(
        "logs", train_cfg.runner.experiment_name,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    log_fp = open(log_path, "w")
    print(f"Logs saved to {log_dir}")

    max_iterations = train_cfg.runner.max_iterations
    save_interval = train_cfg.runner.save_interval
    total_updates = max_iterations

    # ---- 训练循环 ----
    obs, g_vals, h_vals, energy = env.reset()
    obs = obs.to(device)
    g_vals = g_vals.to(device)
    h_vals = h_vals.to(device)
    energy = energy.to(device)

    interval_start = time.time()

    for iteration in range(start_iteration, max_iterations):
        # ---- γ 和 entropy 退火 ----
        gamma_reach = EC_EFPPO.compute_gamma_reach(
            train_cfg.algorithm.gamma_reach_init,
            train_cfg.algorithm.gamma_reach_final,
            iteration,
            total_updates,
        )
        entropy_coef = EC_EFPPO.compute_entropy_coef(
            train_cfg.algorithm.entropy_coef,
            iteration,
            total_updates,
            anneal=train_cfg.algorithm.anneal_entropy,
        )

        # ---- Rollout ----
        for step in range(horizon):
            actions, log_probs, values_energy, values_reach = alg.act(obs)

            next_obs, next_g, next_h, dones, infos, next_energy, energy_consumption = env.step(actions)

            next_obs = next_obs.to(device)
            next_g = next_g.to(device)
            next_h = next_h.to(device)
            next_energy = next_energy.to(device)
            energy_consumption = energy_consumption.to(device)
            dones = dones.to(device)

            # 存储到 buffer
            alg.buffer.add(
                obs=obs,
                actions=actions,
                log_probs=log_probs,
                values=values_energy,
                value_reach=values_reach,
                energy=energy,
                energy_consumption=energy_consumption,
                g_values=g_vals,
                h_values=h_vals,
                dones=dones,
                next_obs=next_obs,
                next_energy=next_energy,
                next_g=next_g,
                next_h=next_h,
            )

            obs = next_obs
            g_vals = next_g
            h_vals = next_h
            energy = next_energy

        # ---- Bootstrap values ----
        with torch.no_grad():
            last_values_energy, last_values_reach = actor_critic.evaluate(obs)

        # ---- 计算优势 ----
        alg.buffer.compute_advantages(
            last_values_energy,
            last_values_reach,
            gamma_energy=train_cfg.algorithm.gamma_energy,
            gamma_reach=gamma_reach,
            gae_lambda=train_cfg.algorithm.gae_lambda,
            gamma_reach_init=train_cfg.algorithm.gamma_reach_init,
            energy_target_rms=alg.energy_target_rms,
        )

        # ---- 成功率和能量消耗 ----
        success_rate, execution_cost, avg_energy = compute_reach_avoid_success_rate(
            alg.buffer.g_values[1:],       # [T, N] g 序列（跳过初始状态）
            alg.buffer.h_values[1:],       # [T, N] h 序列（跳过初始状态）
            energy_sequence=alg.buffer.energy,  # [T+1, N] 完整能量序列
        )
        debug_stats = dict(getattr(alg.buffer, 'debug_stats', {}))

        # ---- 三路 PPO 更新 ----
        loss_dict = alg.update(
            gamma_reach=gamma_reach,
            entropy_coef=entropy_coef,
        )

        # ---- 日志 ----
        if (iteration + 1) % 1 == 0:
            elapsed = time.time() - interval_start
            log_line = (
                f"iter {iteration + 1:05d} | "
                f"success {success_rate:.3f} | "
                f"cost {execution_cost:.1f} | "
                f"energy {avg_energy:.1f} | "
                f"actor_loss {loss_dict['actor_loss']:.5f} | "
                f"energy_loss {loss_dict['energy_loss']:.5f} | "
                f"reach_loss {loss_dict['reach_loss']:.5f} | "
                f"entropy {loss_dict['entropy_loss']:.4f} | "
                f"gamma_reach {gamma_reach:.6f} | "
                f"ent_coef {entropy_coef:.5f} | "
                f"elapsed {elapsed:.2f}s"
            )
            print(log_line)
            log_fp.write(log_line + "\n")

            debug_interval = getattr(train_cfg.runner, 'debug_stats_interval', 0)
            if debug_interval and (iteration + 1) % debug_interval == 0 and debug_stats:
                std = actor_critic.std.detach().float()
                debug_line = (
                    f"debug {iteration + 1:05d} | "
                    f"std_mean {std.mean().item():.4f} | std_min {std.min().item():.4f} | std_max {std.max().item():.4f} | "
                    f"done_mean {debug_stats.get('done_for_gae_mean', float('nan')):.4f} | "
                    f"energy_min_ratio {debug_stats.get('energy_min_ratio', float('nan')):.4f} | "
                    f"energy_neg_ratio {debug_stats.get('energy_negative_ratio', float('nan')):.4f} | "
                    f"v_reach [{debug_stats.get('values_reach_min', float('nan')):.3e}, {debug_stats.get('values_reach_max', float('nan')):.3e}] | "
                    f"t_reach [{debug_stats.get('targets_reach_min', float('nan')):.3e}, {debug_stats.get('targets_reach_max', float('nan')):.3e}] | "
                    f"adv_total_std {debug_stats.get('advantages_total_std', float('nan')):.3e}"
                )
                print(debug_line)
                log_fp.write(debug_line + "\n")

            log_fp.flush()
            interval_start = time.time()

        # ---- 保存 checkpoint ----
        if (iteration + 1) % save_interval == 0:
            save_path = os.path.join(log_dir, f"model_{iteration + 1}.pt")
            # 获取归一化统计量状态
            obs_rms_state = env.env.high_level_env.get_obs_rms_state() if hasattr(env.env, 'high_level_env') else {}
            energy_target_rms_state = alg.get_energy_target_rms_state()
            torch.save(
                {
                    "actor_critic": actor_critic.state_dict(),
                    "policy_optimizer": alg.policy_optimizer.state_dict(),
                    "energy_optimizer": alg.energy_optimizer.state_dict(),
                    "reach_optimizer": alg.reach_optimizer.state_dict(),
                    "iteration": iteration + 1,
                    "success_rate": success_rate,
                    "execution_cost": execution_cost,
                    "avg_energy_consumption": avg_energy,
                    "low_level_model_path": train_cfg.runner.low_level_model_path,
                    "obs_rms_state": obs_rms_state,
                    "energy_target_rms_state": energy_target_rms_state,
                },
                save_path,
            )
            print(f"  saved checkpoint: {save_path}")

        # ---- 重置环境（下一轮 rollout 开始前）----
        if iteration + 1 < max_iterations:
            obs, g_vals, h_vals, energy = env.reset()
            obs = obs.to(device)
            g_vals = g_vals.to(device)
            h_vals = h_vals.to(device)
            energy = energy.to(device)

    # ---- 保存最终模型 ----
    final_path = os.path.join(log_dir, "model_final.pt")
    obs_rms_state = env.env.high_level_env.get_obs_rms_state() if hasattr(env.env, 'high_level_env') else {}
    energy_target_rms_state = alg.get_energy_target_rms_state()
    torch.save(
        {
            "actor_critic": actor_critic.state_dict(),
            "policy_optimizer": alg.policy_optimizer.state_dict(),
            "energy_optimizer": alg.energy_optimizer.state_dict(),
            "reach_optimizer": alg.reach_optimizer.state_dict(),
            "iteration": max_iterations,
            "success_rate": success_rate,
            "avg_energy_consumption": avg_energy,
            "low_level_model_path": train_cfg.runner.low_level_model_path,
            "obs_rms_state": obs_rms_state,
            "energy_target_rms_state": energy_target_rms_state,
        },
        final_path,
    )
    print(f"training complete. final checkpoint: {final_path}")

    log_fp.close()
    env.close()


if __name__ == "__main__":
    args = get_args()
    args.headless = True
    args.compute_device_id = 0
    args.sim_device_id = 0
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    train_ecfppo(args)
