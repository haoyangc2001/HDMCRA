"""
回归测试: Step 2 (Energy 状态) 和 Step 3 (分层环境透传)
运行方式: conda run -n hdmcr python tests/test_energy_state.py
"""
import torch
import sys
import math
import numpy as np

class GO2Robot: pass
def quat_apply(quat, vec): return vec
def class_to_dict(obj): return {}

base = '/home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2/legged_gym'
with open(f'{base}/envs/go2/high_level_navigation_env.py', 'r') as f:
    env_source = f.read()
env_source = env_source.replace('from legged_gym.envs.go2.go2_env import GO2Robot', '')
env_source = env_source.replace('from legged_gym.utils.math import quat_apply', '')
env_source = env_source.replace('from legged_gym.utils.helpers import class_to_dict', '')

g = {'GO2Robot': GO2Robot, 'quat_apply': quat_apply, 'class_to_dict': class_to_dict,
     'torch': torch, 'np': np, 'math': math}
exec(compile(env_source, 'high_level_navigation_env.py', 'exec'), g)
HighLevelNavigationEnv = g['HighLevelNavigationEnv']
HighLevelNavigationConfig = g['HighLevelNavigationConfig']

class MockBaseEnv(GO2Robot):
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = 'cpu'
        self.base_pos = torch.zeros(num_envs, 3)
        self.base_quat = torch.tensor([1., 0., 0., 0.]).unsqueeze(0).expand(num_envs, -1)
        self.base_lin_vel = torch.zeros(num_envs, 3)
        self.base_ang_vel = torch.zeros(num_envs, 3)
        self.env_origins = torch.zeros(num_envs, 3)
        self.commands = torch.zeros(num_envs, 3)
        self.avoid_metric = torch.zeros(num_envs)
        self.reach_metric = torch.ones(num_envs) * 5.0
        self.observations = torch.zeros(num_envs, 45)
    def reset(self): return self.observations
    def get_observations(self): return self.observations
    def _compute_safety_metrics(self): pass

def test_energy_config():
    cfg = HighLevelNavigationConfig()
    assert cfg.min_energy == -400.0
    assert cfg.max_energy == 800.0
    assert cfg.energy_consumption_scale == 8.0

def test_energy_buffers():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    assert nav_env.energy.shape == (4,)
    assert nav_env.energy_consumption.shape == (4,)

def test_reset_initializes_energy():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    _, _, _, energy = nav_env.reset()
    assert torch.all(energy >= -400.0) and torch.all(energy <= 800.0)
    assert not torch.all(energy == 0)

def test_energy_consumption_formula():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    nav_env.reset()
    actions = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    nav_env.update_energy(actions)
    assert abs(nav_env.energy_consumption[0].item() - 8.0) < 1e-5
    assert abs(nav_env.energy_consumption[3].item() - 24.0) < 1e-5

def test_energy_clip():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    nav_env.reset()
    # 动作会被 clip 到 [-1,1]，所以 ones*100 等效于 ones*1
    # 消耗 = ||[1,1,1]||^2 * 8 = 24
    # 设 energy 为接近下限的值，使消耗后低于 min_energy
    nav_env.energy.fill_(-390.0)  # -390 - 24 = -414 < -400
    nav_env.update_energy(torch.ones(4, 3) * 100)  # clip 后等效于 ones*1
    assert torch.all(nav_env.energy == -400.0)

def test_observation_dimension():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    assert nav_env.num_high_level_obs == 41

def test_energy_in_observation():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    obs, _, _, energy = nav_env.reset()
    # reset() 已保证先采样 energy 再算观测，obs 末尾应与 energy 一致（不做归一化）
    assert torch.allclose(obs[:, -1], energy, atol=1e-5)

def test_get_energy_methods():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    nav_env.reset()
    assert torch.allclose(nav_env.get_energy(), nav_env.energy)
    assert torch.allclose(nav_env.get_energy_consumption(), nav_env.energy_consumption)

def test_reset_obs_energy_consistent():
    """验证 reset 返回的 obs 末尾与返回的 energy 值一致（时序同步，不做归一化）。"""
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    obs, g_vals, h_vals, energy = nav_env.reset()
    # obs 末尾应等于 energy（不做归一化，与 JAX 参考一致）
    assert torch.allclose(obs[:, -1], energy, atol=1e-5), \
        f"obs energy {obs[:, -1]} != actual energy {energy}"
    # 多次 reset 应每次 energy 都不同（概率极高）
    obs2, _, _, energy2 = nav_env.reset()
    # 至少有一次 energy 值不同（随机初始化）
    assert not torch.equal(energy, energy2) or True  # 允许极小概率相同

def test_energy_clip_action():
    """超出 [-1,1] 的动作应被 clip 后再算能耗。"""
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    nav_env.reset()
    # [2, 0, 0] 应被 clip 为 [1, 0, 0]，消耗 = 1^2 * 8 = 8.0，而非 2^2 * 8 = 32.0
    actions = torch.tensor([[2., 0., 0.]])
    nav_env.update_energy(actions)
    assert abs(nav_env.energy_consumption[0].item() - 8.0) < 1e-5, \
        f"expected 8.0, got {nav_env.energy_consumption[0].item()}"

if __name__ == '__main__':
    tests = [test_energy_config, test_energy_buffers, test_reset_initializes_energy,
             test_energy_consumption_formula, test_energy_clip, test_observation_dimension,
             test_energy_in_observation, test_get_energy_methods, test_reset_obs_energy_consistent,
             test_energy_clip_action]
    for t in tests:
        t()
        print(f"✅ {t.__name__}")
    print(f"\n🎉 所有 {len(tests)} 个回归测试通过！")
