#!/usr/bin/env python3
"""
测试不同 energy_consumption_scale 对训练效果的影响。

测试配置：
- scale=8.0: 当前默认值（每步最大消耗 120）
- scale=1.0: 降低 8 倍（每步最大消耗 15）
- scale=0.5: 降低 16 倍（每步最大消耗 7.5）

运行方式：
  conda run -n hdmcr python tests/test_energy_consumption_scale.py
"""

import os
import sys
import json
import time
from datetime import datetime

# 设置环境
os.chdir('/home/caohy/repositories/HDMCRA/HDMCRA')
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA')

import isaacgym
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2EC_EFPPOCfgPPO
from legged_gym.scripts.train_ecfppo import HierarchicalVecEnv, create_env, compute_reach_avoid_success_rate
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args

from rsl_rl.algorithms.ecfppo import EC_EFPPO
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


def run_training_with_scale(energy_consumption_scale, max_iterations=50, seed=1):
    """使用指定的 energy_consumption_scale 运行训练。"""

    print(f"\n{'='*60}")
    print(f"Testing energy_consumption_scale = {energy_consumption_scale}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*60}")

    # 创建环境配置
    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2EC_EFPPOCfgPPO()

    # 设置 energy_consumption_scale
    env_cfg.env.energy_consumption_scale = energy_consumption_scale

    # 设置训练参数
    env_cfg.env.num_envs = 256  # 使用较少的 envs 加速测试
    train_cfg.runner.max_iterations = max_iterations
    train_cfg.runner.save_interval = 1000  # 不保存 checkpoint

    # 创建环境
    sim_args = get_args()
    sim_args.headless = True
    sim_args.num_envs = 256
    device = torch.device(sim_args.rl_device)

    env = create_env(env_cfg, train_cfg, sim_args, device)

    # 创建网络
    actor_critic = EC_EFPPO_ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_obs,
        num_actions=env.num_actions,
        hidden_dim=train_cfg.network.hidden_dim,
        num_hidden_layers=train_cfg.network.num_hidden_layers,
        init_noise_std=1.0,
        activation=train_cfg.network.activation,
    ).to(device)

    # 创建算法
    alg = EC_EFPPO(
        actor_critic=actor_critic,
        learning_rate=train_cfg.algorithm.learning_rate,
        gamma_energy=train_cfg.algorithm.gamma_energy,
        gamma_reach_init=train_cfg.algorithm.gamma_reach_init,
        gamma_reach_final=train_cfg.algorithm.gamma_reach_final,
        gae_lambda=train_cfg.algorithm.gae_lambda,
        num_learning_epochs=train_cfg.algorithm.num_learning_epochs,
        num_mini_batches=train_cfg.algorithm.num_mini_batches,
        clip_param=train_cfg.algorithm.clip_param,
        value_loss_coef=train_cfg.algorithm.vf_coef,
        entropy_coef=train_cfg.algorithm.entropy_coef,
        max_grad_norm=train_cfg.algorithm.max_grad_norm,
        device=device,
    )

    # 初始化存储
    alg.init_storage(
        num_envs=env.num_envs,
        horizon=train_cfg.runner.horizon_length,
        obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
    )

    # 训练循环
    obs, g_vals, h_vals, energy = env.reset()

    results = {
        'scale': energy_consumption_scale,
        'iterations': [],
        'success_rates': [],
        'energy_means': [],
        'energy_losses': [],
        'reach_losses': [],
    }

    gamma_reach = train_cfg.algorithm.gamma_reach_init

    for iteration in range(max_iterations):
        # Rollout
        for step in range(train_cfg.runner.horizon_length):
            with torch.no_grad():
                actions, log_probs, values_energy, values_reach = alg.act(obs)

            obs_new, g_vals_new, h_vals_new, dones, infos, energy_new, energy_consumption = env.step(actions)

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
                next_obs=obs_new.to(device),
                next_energy=energy_new,
                next_g=g_vals_new,
                next_h=h_vals_new,
            )

            obs = obs_new.to(device)
            g_vals = g_vals_new
            h_vals = h_vals_new
            energy = energy_new

        # Bootstrap
        with torch.no_grad():
            last_values_energy, last_values_reach = actor_critic.evaluate(obs)

        # 计算优势
        alg.buffer.compute_advantages(
            last_values_energy=last_values_energy,
            last_values_reach=last_values_reach,
            gamma_energy=alg.gamma_energy,
            gamma_reach=gamma_reach,
            gae_lambda=alg.gae_lambda,
            gamma_reach_init=alg.gamma_reach_init,
        )

        # 更新
        loss_dict = alg.update(gamma_reach=gamma_reach, entropy_coef=alg.entropy_coef)

        # 计算成功率
        success_rate, execution_cost, avg_energy = compute_reach_avoid_success_rate(
            alg.buffer.g_values[1:],
            alg.buffer.h_values[1:],
            alg.buffer.energy[1:],
        )

        # 记录结果
        results['iterations'].append(iteration + 1)
        results['success_rates'].append(success_rate)
        results['energy_means'].append(energy.mean().item())
        results['energy_losses'].append(loss_dict.get('energy_loss', 0))
        results['reach_losses'].append(loss_dict.get('reach_loss', 0))

        # 更新 gamma_reach
        gamma_reach = min(
            train_cfg.algorithm.gamma_reach_final,
            train_cfg.algorithm.gamma_reach_init +
            (train_cfg.algorithm.gamma_reach_final - train_cfg.algorithm.gamma_reach_init) *
            (iteration + 1) * 2 / train_cfg.runner.max_iterations
        )

        # 打印进度
        if (iteration + 1) % 10 == 0:
            print(f"  iter {iteration+1:3d} | success {success_rate:.3f} | "
                  f"energy_mean {energy.mean():.1f} | "
                  f"energy_loss {loss_dict.get('energy_loss', 0):.1f} | "
                  f"reach_loss {loss_dict.get('reach_loss', 0):.1f}")

    env.close()
    return results


def main():
    print("="*60)
    "Energy Consumption Scale 对比测试"
    print("="*60)

    # 测试配置
    scales = [8.0, 2.0, 1.0, 0.5]
    max_iterations = 50

    all_results = []

    for scale in scales:
        try:
            result = run_training_with_scale(
                energy_consumption_scale=scale,
                max_iterations=max_iterations,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  Error with scale={scale}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")

    print(f"\n{'Scale':>8} | {'Success Mean':>12} | {'Success Peak':>12} | {'Energy Final':>12} | {'Energy Loss':>12}")
    print("-"*70)

    for r in all_results:
        scale = r['scale']
        success_mean = sum(r['success_rates']) / len(r['success_rates']) if r['success_rates'] else 0
        success_peak = max(r['success_rates']) if r['success_rates'] else 0
        energy_final = r['energy_means'][-1] if r['energy_means'] else 0
        energy_loss_final = r['energy_losses'][-1] if r['energy_losses'] else 0

        print(f"{scale:>8.1f} | {success_mean:>12.4f} | {success_peak:>12.4f} | {energy_final:>12.1f} | {energy_loss_final:>12.1f}")

    # 保存结果
    output_file = 'logs/energy_scale_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")

    # 分析
    print(f"\n{'='*60}")
    print("分析")
    print(f"{'='*60}")

    if len(all_results) >= 2:
        baseline = all_results[0]  # scale=8.0
        for r in all_results[1:]:
            scale = r['scale']
            baseline_success = sum(baseline['success_rates']) / len(baseline['success_rates'])
            test_success = sum(r['success_rates']) / len(r['success_rates'])

            if test_success > baseline_success * 1.2:
                print(f"  scale={scale}: ✅ 成功率提升 {(test_success/baseline_success - 1)*100:.1f}%")
            elif test_success < baseline_success * 0.8:
                print(f"  scale={scale}: ❌ 成功率下降 {(1 - test_success/baseline_success)*100:.1f}%")
            else:
                print(f"  scale={scale}: ➖ 成功率变化不大")


if __name__ == "__main__":
    main()
