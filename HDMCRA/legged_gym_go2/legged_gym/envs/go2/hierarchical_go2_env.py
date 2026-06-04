import copy
import os
from typing import Optional

import torch
from legged_gym.envs.go2.go2_env import GO2Robot
from legged_gym.envs.go2.high_level_navigation_env import HighLevelNavigationEnv, HighLevelNavigationConfig
from legged_gym.utils import task_registry, update_class_from_dict
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner


class HierarchicalGO2Env:
    """
    Hierarchical GO2 environment that combines low-level locomotion control with a high-level navigation policy.
    - Low level: pretrained locomotion policy (velocity commands -> joint actions)
    - High level: navigation policy to be trained (observations -> velocity commands)
    """

    def __init__(self, cfg, low_level_model_path: str, args=None, device='cuda:0'):
        self.cfg = cfg
        self.args = args
        self.device = device
        self.low_level_model_path = low_level_model_path

        self.base_env = self._create_base_env()
        self.low_level_policy = self._load_low_level_policy()

        self.high_level_config = HighLevelNavigationConfig()
        self._update_high_level_config()
        self.high_level_env = HighLevelNavigationEnv(self.base_env, self.high_level_config)
        self.low_level_action_repeat = getattr(self.cfg.env, 'high_level_action_repeat', 1)

        self.num_envs = self.base_env.num_envs
        self.num_obs = self.high_level_env.num_high_level_obs
        self.num_actions = self.high_level_env.num_high_level_actions
        self.device = self.base_env.device

    def _build_base_env_cfg(self):
        """Build the low-level GO2 config without leaking high-level action/obs overrides."""
        base_env_cfg, _ = task_registry.get_cfgs(name="go2")
        merged_cfg = copy.deepcopy(base_env_cfg)

        # Apply runtime overrides that should affect the underlying locomotion env.
        if hasattr(self.cfg, 'seed'):
            merged_cfg.seed = self.cfg.seed

        if hasattr(self.cfg, 'env') and hasattr(self.cfg.env, 'num_envs'):
            merged_cfg.env.num_envs = self.cfg.env.num_envs

        if hasattr(self.cfg, 'terrain'):
            update_class_from_dict(merged_cfg.terrain, class_to_dict(self.cfg.terrain))

        if hasattr(self.cfg, 'commands'):
            update_class_from_dict(merged_cfg.commands, class_to_dict(self.cfg.commands))

        if hasattr(self.cfg, 'control'):
            update_class_from_dict(merged_cfg.control, class_to_dict(self.cfg.control))

        if hasattr(self.cfg, 'domain_rand'):
            update_class_from_dict(merged_cfg.domain_rand, class_to_dict(self.cfg.domain_rand))

        if hasattr(self.cfg, 'asset'):
            update_class_from_dict(merged_cfg.asset, class_to_dict(self.cfg.asset))

        if hasattr(self.cfg, 'rewards'):
            update_class_from_dict(merged_cfg.rewards, class_to_dict(self.cfg.rewards))

        if hasattr(self.cfg, 'rewards_ext'):
            update_class_from_dict(merged_cfg.rewards_ext, class_to_dict(self.cfg.rewards_ext))

        if hasattr(self.cfg, 'sim'):
            update_class_from_dict(merged_cfg.sim, class_to_dict(self.cfg.sim))

        return merged_cfg

    def _create_base_env(self):
        """Instantiate the original GO2 environment using the merged low-level env_cfg."""
        base_env_cfg = self._build_base_env_cfg()
        env, _ = task_registry.make_env(name="go2", args=self.args, env_cfg=base_env_cfg)
        return env

    def _get_dummy_args(self):
        if self.args is not None:
            return self.args

        class DummyArgs:
            def __init__(self, device, cfg):
                self.headless = False
                self.rl_device = str(device)
                self.sim_device = str(device)
                self.graphics_device_id = 0
                self.num_envs = getattr(cfg.env, 'num_envs', 2)
                self.physics_engine = "physx"
                self.use_gpu = True
                self.use_gpu_pipeline = True
                self.subscenes = 0
                self.num_threads = 0
                self.sim_device_type = "cuda" if "cuda" in str(device) else "cpu"
                self.compute_device_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
                self.sim_device_id = self.compute_device_id

        return DummyArgs(self.device, self.cfg)

    def _load_low_level_policy(self):
        if not os.path.exists(self.low_level_model_path):
            raise FileNotFoundError(f"Low-level policy checkpoint not found: {self.low_level_model_path}")

        train_cfg = self._get_low_level_train_cfg()
        train_cfg_dict = class_to_dict(train_cfg)
        ppo_runner = OnPolicyRunner(self.base_env, train_cfg_dict, device=self.device)

        print(f"Loading low-level policy checkpoint: {self.low_level_model_path}")
        ppo_runner.load(self.low_level_model_path)
        return ppo_runner.get_inference_policy(device=self.device)

    def _get_low_level_train_cfg(self):
        _, train_cfg = task_registry.get_cfgs(name="go2")
        train_cfg.runner.resume = False
        return train_cfg

    def _update_high_level_config(self):
        """Copy all relevant high-level parameters from env_cfg."""
        if hasattr(self.cfg, 'rewards_ext'):
            if hasattr(self.cfg.rewards_ext, 'unsafe_spheres_pos'):
                self.high_level_config.unsafe_spheres_pos = self.cfg.rewards_ext.unsafe_spheres_pos
            if hasattr(self.cfg.rewards_ext, 'unsafe_sphere_pos'):
                self.high_level_config.unsafe_sphere_pos = self.cfg.rewards_ext.unsafe_sphere_pos
            if hasattr(self.cfg.rewards_ext, 'unsafe_sphere_radius'):
                self.high_level_config.unsafe_sphere_radius = self.cfg.rewards_ext.unsafe_sphere_radius
            if hasattr(self.cfg.rewards_ext, 'target_sphere_radius'):
                self.high_level_config.target_radius = self.cfg.rewards_ext.target_sphere_radius
            if hasattr(self.cfg.rewards_ext, 'target_sphere_pos'):
                self.high_level_config.target_sphere_pos = self.cfg.rewards_ext.target_sphere_pos
            if hasattr(self.cfg.rewards_ext, 'boundary_margin'):
                self.high_level_config.boundary_margin = self.cfg.rewards_ext.boundary_margin
        for attr in (
            'enable_manual_lidar',
            'lidar_max_range',
            'lidar_num_bins',
            'target_lidar_num_bins',
            'target_lidar_max_range',
            'boundary_half_extents',
            'action_scale',
            'min_energy',
            'max_energy',
            'energy_consumption_scale',
        ):
            if hasattr(self.cfg, attr):
                setattr(self.high_level_config, attr, getattr(self.cfg, attr))

    def reset(self):
        high_level_obs, g_values, h_values, energy = self.high_level_env.reset()
        return high_level_obs, g_values, h_values, energy

    def step(self, high_level_actions):
        self.high_level_env.update_energy(high_level_actions, repeat=self.low_level_action_repeat)
        self.high_level_env.update_velocity_commands(high_level_actions)
        desired_velocity_commands = self.base_env.commands[:, :3].clone()

        base_obs = None
        base_infos = None
        avoid_metric = None
        reach_metric = None
        aggregated_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.low_level_action_repeat):
            self.base_env.commands[:, :3] = desired_velocity_commands
            self.base_env.compute_observations()
            current_base_obs = self.base_env.get_observations()
            with torch.no_grad():
                low_level_actions = self.low_level_policy(current_base_obs)

            base_obs, privileged_obs, _, step_dones, base_infos, avoid_metric, reach_metric = self.base_env.step(
                low_level_actions
            )
            aggregated_dones |= step_dones.bool()

        self.high_level_env._compute_high_level_observations()
        high_level_obs = self.high_level_env.get_observations()
        g_values, h_values = self.high_level_env.compute_g_h_values(avoid_metric, reach_metric)
        energy = self.high_level_env.get_energy()
        energy_consumption = self.high_level_env.get_energy_consumption()

        infos = {
            'base_infos': base_infos,
            'avoid_metric': avoid_metric,
            'reach_metric': reach_metric,
            'base_obs': base_obs
        }

        return high_level_obs, g_values, h_values, aggregated_dones, infos, energy, energy_consumption

    def get_observations(self):
        return self.high_level_env.get_observations()

    def close(self):
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


def create_hierarchical_go2_env(env_cfg, low_level_model_path: str, device='cuda:0'):
    return HierarchicalGO2Env(env_cfg, low_level_model_path, device)
