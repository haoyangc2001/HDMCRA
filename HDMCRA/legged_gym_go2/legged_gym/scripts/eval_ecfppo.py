#!/usr/bin/env python3
"""
EC-EFPPO 行为级评估脚本。

加载训练好的 EC-EFPPO checkpoint，在仿真中运行策略，
收集轨迹、成功/失败统计、能耗数据，用于行为级评估。

用法：
  python legged_gym/scripts/eval_ecfppo.py \
    --checkpoint-path logs/ecfppo_go2/20260604-105543/model_100.pt \
    --num-trajs 20 --max-steps 200 --render
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Dict, Any

import isaacgym
from isaacgym import gymtorch
import torch
from legged_gym.envs.go2.go2_config import GO2HighLevelCfg, GO2EC_EFPPOCfgPPO
from legged_gym.scripts.train_ecfppo import HierarchicalVecEnv, create_env
from legged_gym.utils import get_args
from legged_gym.utils.helpers import update_cfg_from_args
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic

# 固定的测试起始点，覆盖四个象限
FIXED_SPAWN_POINTS = [(-5, 5), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0)]


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EC-EFPPO 行为级评估：加载 checkpoint，运行 rollout，收集行为数据。"
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to the trained EC-EFPPO checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-trajs",
        type=int,
        default=20,
        help="How many trajectories to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max high-level steps per trajectory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ecfppo_eval_results.json",
        help="JSON file to store evaluation results.",
    )
    parser.add_argument(
        "--low-level-model",
        type=str,
        default=None,
        help="Optional override for the pretrained low-level locomotion policy.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel envs for rollouts (each env produces one trajectory).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open Isaac Gym viewer during rollouts.",
    )
    parser.add_argument(
        "--max-reset-attempts",
        type=int,
        default=30,
        help="Safety cap on how many environment resets we attempt.",
    )
    eval_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if eval_args.num_envs != 1:
        print("[INFO] For sequential trajectory collection we force num_envs=1.")
    eval_args.num_envs = 1
    return eval_args


def _resolve_low_level_model_path(
    eval_args: argparse.Namespace,
    train_cfg: GO2EC_EFPPOCfgPPO,
    checkpoint: dict,
) -> str:
    candidates = []
    if eval_args.low_level_model:
        candidates.append(("argument --low-level-model", eval_args.low_level_model))
    ckpt_path = checkpoint.get("low_level_model_path") if isinstance(checkpoint, dict) else None
    if ckpt_path:
        candidates.append(("checkpoint metadata", ckpt_path))
    cfg_path = getattr(train_cfg.runner, "low_level_model_path", None)
    if cfg_path:
        candidates.append(("config default", cfg_path))

    checked_sources = []
    for source, path in candidates:
        expanded = os.path.expanduser(str(path))
        checked_sources.append(f"{source}: {expanded}")
        if os.path.isfile(expanded):
            print(f"Using low-level policy from {source}: {expanded}")
            return expanded
        else:
            print(f"[WARN] Low-level checkpoint not found at {expanded} ({source}).")

    detail = "\n  ".join(checked_sources) if checked_sources else "  (no candidates available)"
    raise FileNotFoundError(
        "Unable to locate a valid low-level policy checkpoint. Checked:\n"
        f"{detail}\n"
        "Please pass --low-level-model with an existing file, or update the config/checkpoint metadata."
    )


def _set_spawn_position(env: HierarchicalVecEnv, spawn_xy: Sequence[float]):
    base_env = env.env.base_env
    high_level_env = env.env.high_level_env
    device = base_env.device
    spawn_tensor = torch.tensor(spawn_xy, device=device, dtype=torch.float)

    base_env.root_states[0, 0] = spawn_tensor[0]
    base_env.root_states[0, 1] = spawn_tensor[1]
    env_id = torch.tensor([0], device=device, dtype=torch.int32)
    base_env.gym.set_actor_root_state_tensor_indexed(
        base_env.sim,
        gymtorch.unwrap_tensor(base_env.root_states),
        gymtorch.unwrap_tensor(env_id),
        len(env_id),
    )
    base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
    base_env.compute_observations()
    high_level_env._compute_high_level_observations()
    obs = high_level_env.get_observations()
    avoid_metric, reach_metric = high_level_env._get_current_metrics()
    g_vals, h_vals = high_level_env.compute_g_h_values(avoid_metric, reach_metric)
    return obs, g_vals, h_vals


def build_env_and_policy(eval_args: argparse.Namespace):
    """Create the hierarchical env and load the trained EC-EFPPO policy."""
    sim_args = get_args()
    sim_args.headless = not eval_args.render
    sim_args.num_envs = eval_args.num_envs
    device = torch.device(sim_args.rl_device)

    checkpoint = torch.load(eval_args.checkpoint_path, map_location="cpu")

    env_cfg = GO2HighLevelCfg()
    train_cfg = GO2EC_EFPPOCfgPPO()
    env_cfg.env.num_envs = eval_args.num_envs
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, sim_args)
    resolved_low_level = _resolve_low_level_model_path(eval_args, train_cfg, checkpoint)
    train_cfg.runner.low_level_model_path = resolved_low_level

    env = create_env(env_cfg, train_cfg, sim_args, device)

    # 加载归一化统计量（如果存在）
    obs_rms_state = checkpoint.get("obs_rms_state", {})
    if obs_rms_state and hasattr(env.env, 'high_level_env'):
        env.env.high_level_env.set_obs_rms_state(obs_rms_state)
        env.env.high_level_env.set_training(False)  # 评估模式，不更新统计量
        print(f"Loaded observation normalization stats from checkpoint")

    # 从 checkpoint 推断网络结构
    state_dict = checkpoint.get("actor_critic", {})
    # 检查是否是三网络结构（EC_EFPPO_ActorCritic）
    is_ecfppo = any(k.startswith("actor.") for k in state_dict.keys())

    if is_ecfppo:
        # 从 state_dict 推断 hidden_dim 和 num_hidden_layers
        hidden_dim = None
        num_hidden_layers = None
        for key in state_dict:
            if key.startswith("actor.") and key.endswith(".weight"):
                parts = key.split(".")
                layer_idx = int(parts[1])
                if layer_idx == 0:
                    hidden_dim = state_dict[key].shape[0]
                # 计算层数
        if hidden_dim is not None:
            # 数 actor 中的 Linear 层
            linear_count = sum(1 for k in state_dict if k.startswith("actor.") and k.endswith(".weight"))
            num_hidden_layers = linear_count - 1  # 减去输出层
            print(f"Inferred from checkpoint: hidden_dim={hidden_dim}, num_hidden_layers={num_hidden_layers}")

        actor_critic = EC_EFPPO_ActorCritic(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_obs,
            num_actions=env.num_actions,
            hidden_dim=hidden_dim or train_cfg.network.hidden_dim,
            num_hidden_layers=num_hidden_layers or train_cfg.network.num_hidden_layers,
            init_noise_std=1.0,
            activation=train_cfg.network.activation,
        ).to(device)
    else:
        raise ValueError(
            "Checkpoint does not contain EC_EFPPO_ActorCritic state_dict "
            "(expected keys starting with 'actor.'). "
            "Use test_reach_avoid.py for standard ActorCritic checkpoints."
        )

    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()

    # 打印 checkpoint 元信息
    print(f"\n{'='*60}")
    print(f"Checkpoint: {eval_args.checkpoint_path}")
    print(f"  Iteration: {checkpoint.get('iteration', 'N/A')}")
    print(f"  Success Rate: {checkpoint.get('success_rate', 'N/A')}")
    print(f"  Execution Cost: {checkpoint.get('execution_cost', 'N/A')}")
    print(f"  Avg Energy: {checkpoint.get('avg_energy_consumption', 'N/A')}")
    print(f"{'='*60}\n")

    return env, actor_critic, device, train_cfg


def _snapshot_xy(env: HierarchicalVecEnv) -> torch.Tensor:
    """Return the current XY base positions for every env."""
    return env.env.base_env.base_pos[:, :2].detach().clone()


def _get_energy(env: HierarchicalVecEnv) -> float:
    """Return current energy for env 0."""
    return float(env.env.high_level_env.energy[0].item())


def _get_g_h(env: HierarchicalVecEnv):
    """Return current g and h values for env 0."""
    high_level_env = env.env.high_level_env
    avoid_metric, reach_metric = high_level_env._get_current_metrics()
    g_vals, h_vals = high_level_env.compute_g_h_values(avoid_metric, reach_metric)
    return float(g_vals[0].item()), float(h_vals[0].item())


def collect_trajectories(
    env: HierarchicalVecEnv,
    actor_critic: EC_EFPPO_ActorCritic,
    device: torch.device,
    num_trajs: int,
    max_steps: int,
    max_reset_attempts: int,
) -> List[Dict[str, Any]]:
    """Roll out and collect detailed trajectory data."""
    recorded: List[Dict[str, Any]] = []
    envs_per_batch = env.num_envs
    resets_attempted = 0
    if num_trajs < 1:
        return recorded

    spawn_sequence: List[Sequence[float]] = []
    while len(spawn_sequence) < num_trajs:
        spawn_sequence.extend(FIXED_SPAWN_POINTS)
    spawn_sequence = spawn_sequence[:num_trajs]

    for traj_idx, spawn_xy in enumerate(spawn_sequence):
        if resets_attempted >= max_reset_attempts:
            break
        resets_attempted += 1

        print(f"\n--- Trajectory {traj_idx+1}/{num_trajs} | Spawn: {spawn_xy} ---")

        obs, _, _, _ = env.reset()
        obs, _, _ = _set_spawn_position(env, spawn_xy)
        obs = obs.to(device)

        # 收集轨迹数据
        trajectory = {
            "traj_id": traj_idx,
            "spawn_xy": list(spawn_xy),
            "positions": [],
            "energies": [],
            "g_values": [],
            "h_values": [],
            "actions": [],
            "reached_goal": False,
            "violated_constraint": False,
            "final_step": 0,
            "final_energy": 0.0,
        }

        xy = _snapshot_xy(env).cpu().numpy()
        trajectory["positions"].append([float(xy[0, 0]), float(xy[0, 1])])

        initial_energy = _get_energy(env)
        trajectory["energies"].append(initial_energy)

        g_init, h_init = _get_g_h(env)
        trajectory["g_values"].append(g_init)
        trajectory["h_values"].append(h_init)

        dones = torch.zeros(envs_per_batch, dtype=torch.bool, device=device)
        for step in range(max_steps):
            with torch.no_grad():
                actions = actor_critic.act_inference(obs)

            obs, g_vals, h_vals, step_dones, infos, energy, energy_consumption = env.step(actions)
            obs = obs.to(device)

            xy = _snapshot_xy(env).cpu().numpy()
            trajectory["positions"].append([float(xy[0, 0]), float(xy[0, 1])])

            current_energy = _get_energy(env)
            trajectory["energies"].append(current_energy)

            g_val = float(g_vals[0].item())
            h_val = float(h_vals[0].item())
            trajectory["g_values"].append(g_val)
            trajectory["h_values"].append(h_val)

            action_vals = actions[0].cpu().numpy().tolist()
            trajectory["actions"].append(action_vals)

            # 检查是否到达目标（g < 0）
            if g_val < 0 and not trajectory["reached_goal"]:
                trajectory["reached_goal"] = True
                trajectory["reached_at_step"] = step + 1
                print(f"  ✓ Reached goal at step {step+1}, energy={current_energy:.1f}")

            # 检查是否违反约束（h >= 0）
            if h_val >= 0 and not trajectory["violated_constraint"]:
                trajectory["violated_constraint"] = True
                trajectory["constraint_violated_at_step"] = step + 1
                print(f"  ✗ Constraint violated at step {step+1}, h={h_val:.3f}")

            dones |= step_dones.to(device).bool()
            if torch.all(dones):
                trajectory["final_step"] = step + 1
                break

        if trajectory["final_step"] == 0:
            trajectory["final_step"] = max_steps

        trajectory["final_energy"] = trajectory["energies"][-1]
        trajectory["energy_consumed"] = trajectory["energies"][0] - trajectory["energies"][-1]

        # 判断成功：先到达目标且到达前未违反约束
        trajectory["success"] = trajectory["reached_goal"] and not trajectory["violated_constraint"]

        if trajectory["success"]:
            print(f"  Result: SUCCESS (step={trajectory.get('reached_at_step')}, "
                  f"energy_consumed={trajectory['energy_consumed']:.1f})")
        elif trajectory["reached_goal"]:
            print(f"  Result: REACHED BUT UNSAFE (constraint violated)")
        else:
            print(f"  Result: FAILED (did not reach goal, "
                  f"energy_consumed={trajectory['energy_consumed']:.1f})")

        recorded.append(trajectory)

    return recorded


def compute_eval_summary(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from trajectory data."""
    n = len(trajectories)
    if n == 0:
        return {"error": "No trajectories collected"}

    successes = [t for t in trajectories if t["success"]]
    reached = [t for t in trajectories if t["reached_goal"]]
    violated = [t for t in trajectories if t["violated_constraint"]]

    summary = {
        "num_trajectories": n,
        "success_count": len(successes),
        "success_rate": len(successes) / n,
        "reached_goal_count": len(reached),
        "reached_goal_rate": len(reached) / n,
        "constraint_violation_count": len(violated),
        "constraint_violation_rate": len(violated) / n,
    }

    if successes:
        summary["avg_success_steps"] = sum(t.get("reached_at_step", 0) for t in successes) / len(successes)
        summary["avg_success_energy_consumed"] = sum(t["energy_consumed"] for t in successes) / len(successes)
        summary["avg_success_final_energy"] = sum(t["final_energy"] for t in successes) / len(successes)

    if reached:
        summary["avg_reached_steps"] = sum(t.get("reached_at_step", 0) for t in reached) / len(reached)

    summary["avg_energy_consumed_all"] = sum(t["energy_consumed"] for t in trajectories) / n
    summary["avg_final_energy_all"] = sum(t["final_energy"] for t in trajectories) / n
    summary["avg_final_step_all"] = sum(t["final_step"] for t in trajectories) / n

    # 分类统计
    summary["outcome_breakdown"] = {
        "success": len(successes),
        "reached_but_unsafe": len(reached) - len(successes),
        "failed_no_reach": n - len(reached),
    }

    return summary


def save_results(
    trajectories: List[Dict[str, Any]],
    summary: Dict[str, Any],
    eval_args: argparse.Namespace,
    train_cfg: GO2EC_EFPPOCfgPPO,
) -> Path:
    out_path = Path(eval_args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 轻量化轨迹数据（去掉详细的逐步 actions 以减小文件大小）
    trajectories_light = []
    for t in trajectories:
        t_light = {k: v for k, v in t.items() if k != "actions"}
        # 只保留首尾和关键节点的 action
        if t["actions"]:
            t_light["action_first"] = t["actions"][0]
            t_light["action_last"] = t["actions"][-1]
        trajectories_light.append(t_light)

    payload = {
        "checkpoint_path": os.path.abspath(eval_args.checkpoint_path),
        "eval_config": {
            "num_trajs": eval_args.num_trajs,
            "max_steps": eval_args.max_steps,
            "low_level_model": train_cfg.runner.low_level_model_path,
        },
        "summary": summary,
        "trajectories": trajectories_light,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    eval_args = parse_eval_args()
    env, actor_critic, device, train_cfg = build_env_and_policy(eval_args)

    try:
        trajectories = collect_trajectories(
            env,
            actor_critic,
            device,
            num_trajs=eval_args.num_trajs,
            max_steps=eval_args.max_steps,
            max_reset_attempts=eval_args.max_reset_attempts,
        )

        summary = compute_eval_summary(trajectories)

        # 打印汇总
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total trajectories: {summary['num_trajectories']}")
        print(f"  Success: {summary['success_count']} ({summary['success_rate']*100:.1f}%)")
        print(f"  Reached goal: {summary['reached_goal_count']} ({summary['reached_goal_rate']*100:.1f}%)")
        print(f"  Constraint violations: {summary['constraint_violation_count']} ({summary['constraint_violation_rate']*100:.1f}%)")
        print(f"\n  Outcome breakdown:")
        for k, v in summary["outcome_breakdown"].items():
            print(f"    {k}: {v}")
        if "avg_success_steps" in summary:
            print(f"\n  Success stats:")
            print(f"    Avg steps to goal: {summary['avg_success_steps']:.1f}")
            print(f"    Avg energy consumed: {summary['avg_success_energy_consumed']:.1f}")
        print(f"\n  Overall stats:")
        print(f"    Avg energy consumed: {summary['avg_energy_consumed_all']:.1f}")
        print(f"    Avg final energy: {summary['avg_final_energy_all']:.1f}")
        print(f"    Avg final step: {summary['avg_final_step_all']:.1f}")
        print(f"{'='*60}")

        out_path = save_results(trajectories, summary, eval_args, train_cfg)
        print(f"\nResults saved to: {out_path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
