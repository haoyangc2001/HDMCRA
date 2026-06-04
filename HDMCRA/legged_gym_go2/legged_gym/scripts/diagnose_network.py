#!/usr/bin/env python3
"""
诊断网络输出和观测值。

检查：
1. 观测值的范围
2. 网络各层的激活值
3. 网络权重的范围
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
    sim_args.num_envs = 4
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

    # 检查网络权重
    print("="*60)
    print("网络权重分析")
    print("="*60)

    for name, param in actor_critic.named_parameters():
        print(f"{name}: shape={param.shape}, min={param.min():.4f}, max={param.max():.4f}, mean={param.mean():.4f}, std={param.std():.4f}")

    # 获取观测值
    obs, g_vals, h_vals, energy = env.reset()
    obs = obs.to(device)

    print(f"\n{'='*60}")
    print("观测值分析")
    print(f"{'='*60}")

    print(f"观测维度: {obs.shape}")
    print(f"观测值范围: min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}")
    print(f"观测值标准差: {obs.std():.4f}")

    # 检查各观测维度
    print(f"\n各观测维度:")
    for i in range(min(10, obs.shape[1])):
        print(f"  维度 {i}: min={obs[:, i].min():.4f}, max={obs[:, i].max():.4f}, mean={obs[:, i].mean():.4f}")

    # 检查网络各层激活值
    print(f"\n{'='*60}")
    print("网络激活值分析")
    print(f"{'='*60}")

    # 注册 hook 来捕获中间激活值
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # 注册 hook 到每一层
    for i, layer in enumerate(actor_critic.actor):
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(get_activation(f'actor_linear_{i}'))
        elif isinstance(layer, torch.nn.ELU) or isinstance(layer, torch.nn.Tanh):
            layer.register_forward_hook(get_activation(f'actor_activation_{i}'))

    # 前向传播
    with torch.no_grad():
        actions, log_probs, energy_value, reach_value = actor_critic.act(obs)

    # 打印激活值
    print(f"\nActor 网络激活值:")
    for name, activation in activations.items():
        print(f"  {name}: shape={activation.shape}, min={activation.min():.4f}, max={activation.max():.4f}, mean={activation.mean():.4f}")

    # 检查输出
    print(f"\n{'='*60}")
    print("网络输出分析")
    print(f"{'='*60}")

    print(f"动作值 (raw): min={actions.min():.4f}, max={actions.max():.4f}, mean={actions.mean():.4f}")
    print(f"动作值 (clipped): min={actions.clamp(-1, 1).min():.4f}, max={actions.clamp(-1, 1).max():.4f}")
    print(f"log_probs: min={log_probs.min():.4f}, max={log_probs.max():.4f}, mean={log_probs.mean():.4f}")
    print(f"energy_value: min={energy_value.min():.4f}, max={energy_value.max():.4f}, mean={energy_value.mean():.4f}")
    print(f"reach_value: min={reach_value.min():.4f}, max={reach_value.max():.4f}, mean={reach_value.mean():.4f}")

    # 检查 std
    print(f"\n{'='*60}")
    print("探索噪声分析")
    print(f"{'='*60}")

    print(f"log_std (learnable): {actor_critic.std.data}")
    print(f"std: {actor_critic.std.data.exp() if hasattr(actor_critic.std.data, 'exp') else actor_critic.std.data}")

    env.close()


if __name__ == "__main__":
    main()
