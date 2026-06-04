#!/usr/bin/env python3
"""
诊断 reach 信号是否在策略移动时正确变化。

检查：
1. reach 值是否随时间变化
2. 策略移动时 reach 是否减少
3. reach advantage 是否提供了有用的梯度
"""

import os
import sys

import isaacgym
import torch

from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2EC_EFPPOCfgPPO
from legged_gym.scripts.train_ecfppo import HierarchicalVecEnv, create_env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


def main():
    # 设置环境
    sim_args = get_args()
    sim_args.headless = True
    sim_args.num_envs = 4  # 使用少量 envs 进行详细分析
    device = torch.device(sim_args.rl_device)

    # 加载 checkpoint
    checkpoint_path = '/home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2/logs/ecfppo_go2/20260604-105543/model_100.pt'
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2EC_EFPPOCfgPPO()
    env_cfg.env.num_envs = 4
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, sim_args)

    if "low_level_model_path" in checkpoint:
        train_cfg.runner.low_level_model_path = checkpoint["low_level_model_path"]

    env = create_env(env_cfg, train_cfg, sim_args, device)

    # 加载策略
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

    # 运行 rollout 并记录 reach 值
    print("="*60)
    print("Reach 信号变化分析")
    print("="*60)

    obs, g_vals, h_vals, energy = env.reset()
    obs = obs.to(device)

    num_steps = 30
    reach_history = []
    energy_history = []
    action_history = []
    position_history = []

    for step in range(num_steps):
        with torch.no_grad():
            actions, log_probs, energy_value, reach_value = actor_critic.act(obs)

        # 记录当前位置
        base_pos = env.env.base_env.base_pos[:, :2].clone()
        position_history.append(base_pos.cpu())

        # 记录 reach 值
        reach_history.append(g_vals.cpu())
        energy_history.append(energy.cpu())
        action_history.append(actions.cpu())

        # 执行动作
        obs_new, g_vals_new, h_vals_new, dones, infos, energy_new, energy_consumption = env.step(actions)

        obs = obs_new.to(device)
        g_vals = g_vals_new
        h_vals = h_vals_new
        energy = energy_new

    # 分析结果
    print(f"\n--- Reach 值变化 ---")
    for env_idx in range(4):
        print(f"\n环境 {env_idx}:")
        for step in range(min(10, num_steps)):
            reach = reach_history[step][env_idx].item()
            energy = energy_history[step][env_idx].item()
            action = action_history[step][env_idx]
            pos = position_history[step][env_idx]
            print(f"  Step {step:2d}: reach={reach:8.2f}, energy={energy:8.1f}, "
                  f"pos=({pos[0]:.2f}, {pos[1]:.2f}), action=({action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f})")

    # 分析 reach 变化趋势
    print(f"\n--- Reach 变化趋势 ---")
    for env_idx in range(4):
        reach_vals = [reach_history[step][env_idx].item() for step in range(num_steps)]
        reach_change = reach_vals[-1] - reach_vals[0]
        print(f"环境 {env_idx}: 初始={reach_vals[0]:.2f}, 最终={reach_vals[-1]:.2f}, 变化={reach_change:.2f}")

    # 分析能量消耗
    print(f"\n--- 能量消耗分析 ---")
    for env_idx in range(4):
        energy_vals = [energy_history[step][env_idx].item() for step in range(num_steps)]
        energy_change = energy_vals[-1] - energy_vals[0]
        print(f"环境 {env_idx}: 初始={energy_vals[0]:.1f}, 最终={energy_vals[-1]:.1f}, 消耗={-energy_change:.1f}")

    # 分析动作模式
    print(f"\n--- 动作模式分析 ---")
    for env_idx in range(4):
        actions = [action_history[step][env_idx] for step in range(num_steps)]
        mean_action = torch.stack(actions).mean(dim=0)
        std_action = torch.stack(actions).std(dim=0)
        print(f"环境 {env_idx}: 均值=({mean_action[0]:.3f}, {mean_action[1]:.3f}, {mean_action[2]:.3f}), "
              f"标准差=({std_action[0]:.3f}, {std_action[1]:.3f}, {std_action[2]:.3f})")

    env.close()


if __name__ == "__main__":
    main()
