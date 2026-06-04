"""
回归测试: Step 7 - EC-EFPPO 训练脚本改造
运行方式: conda run -n hdmcr python tests/test_train_ecfppo.py
"""
import isaacgym
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA/rsl_rl')
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2')

from legged_gym.envs.go2.go2_config import (
    GO2HighLevelCfg, GO2HighLevelCfgPPO, GO2EC_EFPPOCfgPPO
)
from rsl_rl.algorithms.ecfppo import EC_EFPPO
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


# ---- Test 1: GO2EC_EFPPOCfgPPO 配置类 ----
def test_config_class():
    cfg = GO2EC_EFPPOCfgPPO()
    assert cfg.algorithm.gamma_energy == 0.99
    assert cfg.algorithm.gamma_reach_init == 0.999
    assert cfg.algorithm.gamma_reach_final == 0.99999
    assert cfg.algorithm.gae_lambda == 0.95
    assert cfg.algorithm.clip_eps == 0.2
    assert cfg.algorithm.vf_coef == 1.0
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.anneal_entropy == False
    assert cfg.algorithm.max_grad_norm == 0.5
    assert cfg.algorithm.learning_rate == 1e-3
    assert cfg.algorithm.num_learning_epochs == 10
    assert cfg.algorithm.num_mini_batches == 8
    assert cfg.runner.experiment_name == 'ecfppo_go2'
    print("[PASS] test_config_class")


# ---- Test 2: 配置类继承关系 ----
def test_config_inheritance():
    cfg = GO2EC_EFPPOCfgPPO()
    # 应该能访问父类的属性
    assert hasattr(cfg.algorithm, 'entropy_coef')
    assert hasattr(cfg.runner, 'max_iterations')
    assert hasattr(cfg.runner, 'save_interval')
    assert hasattr(cfg.runner, 'low_level_model_path')
    print("[PASS] test_config_inheritance")


# ---- Test 3: EC_EFPPO_ActorCritic 与配置兼容 ----
def test_model_with_config():
    cfg = GO2HighLevelCfg()
    num_obs = cfg.env.num_observations
    num_actions = cfg.env.num_actions

    model = EC_EFPPO_ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        hidden_dim=256,
        num_hidden_layers=2,
    )
    assert model.actor[0].in_features == num_obs
    assert model.actor[-1].out_features == num_actions
    assert model.energy_critic[-1].out_features == 1
    assert model.reach_critic[-1].out_features == 1
    print(f"  num_obs={num_obs}, num_actions={num_actions}")
    print("[PASS] test_model_with_config")


# ---- Test 4: EC_EFPPO 与配置兼容 ----
def test_alg_with_config():
    cfg = GO2EC_EFPPOCfgPPO()
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=48, num_critic_obs=48, num_actions=12,
        hidden_dim=64, num_hidden_layers=2,
    )
    alg = EC_EFPPO(
        actor_critic=model,
        learning_rate=cfg.algorithm.learning_rate,
        gamma_energy=cfg.algorithm.gamma_energy,
        gamma_reach_init=cfg.algorithm.gamma_reach_init,
        gamma_reach_final=cfg.algorithm.gamma_reach_final,
        gae_lambda=cfg.algorithm.gae_lambda,
        num_learning_epochs=cfg.algorithm.num_learning_epochs,
        num_mini_batches=cfg.algorithm.num_mini_batches,
        clip_param=cfg.algorithm.clip_eps,
        value_loss_coef=cfg.algorithm.vf_coef,
        entropy_coef=cfg.algorithm.entropy_coef,
        max_grad_norm=cfg.algorithm.max_grad_norm,
        anneal_entropy=cfg.algorithm.anneal_entropy,
        device='cpu',
    )
    assert alg.gamma_energy == 0.99
    assert alg.gamma_reach_init == 0.999
    assert alg.clip_param == 0.2
    assert alg.value_loss_coef == 1.0
    assert alg.anneal_entropy == False
    print("[PASS] test_alg_with_config")


# ---- Test 5: compute_reach_avoid_success_rate 含能量统计 ----
def test_success_rate_with_energy():
    # 从 train_ecfppo.py 导入
    from legged_gym.scripts.train_ecfppo import compute_reach_avoid_success_rate

    T, N = 20, 8
    # 构造数据：前4个环境成功到达（g < 0），后4个失败
    g_seq = torch.ones(T, N) * 0.5
    g_seq[10:, :4] = -0.5  # 前4个环境在 t=10 到达目标

    h_seq = torch.ones(T, N) * -1.0  # 全部安全（h < 0）

    energy_seq = torch.zeros(T + 1, N)
    for t in range(T + 1):
        energy_seq[t] = 400.0 - t * 5.0  # 每步消耗 5 单位能量

    success_rate, exec_cost, avg_energy = compute_reach_avoid_success_rate(
        g_seq, h_seq, energy_seq
    )

    assert success_rate == 0.5, f"success_rate: {success_rate}"
    assert exec_cost == 10.0, f"exec_cost: {exec_cost}"
    # g_seq[10] 首次到达，对应 energy_seq[11]，消耗 11 * 5 = 55
    assert abs(avg_energy - 55.0) < 1e-4, f"avg_energy: {avg_energy}"
    print(f"  success_rate={success_rate}, exec_cost={exec_cost}, avg_energy={avg_energy}")
    print("[PASS] test_success_rate_with_energy")


# ---- Test 6: compute_reach_avoid_success_rate 无能量统计 ----
def test_success_rate_without_energy():
    from legged_gym.scripts.train_ecfppo import compute_reach_avoid_success_rate

    T, N = 10, 4
    g_seq = torch.ones(T, N) * 0.5
    g_seq[5:, :2] = -0.5
    h_seq = torch.ones(T, N) * -1.0

    success_rate, exec_cost, avg_energy = compute_reach_avoid_success_rate(g_seq, h_seq)
    assert success_rate == 0.5
    assert exec_cost == 5.0
    assert avg_energy == 0.0  # 无 energy_sequence 时返回 0
    print("[PASS] test_success_rate_without_energy")


def test_success_rate_energy_alignment_uses_post_step_state():
    from legged_gym.scripts.train_ecfppo import compute_reach_avoid_success_rate

    T, N = 6, 2
    g_seq = torch.ones(T, N)
    h_seq = -torch.ones(T, N)
    g_seq[2:, 0] = -1.0  # env0 在 t=2 首次到达
    g_seq[4:, 1] = -1.0  # env1 在 t=4 首次到达

    energy_seq = torch.tensor([
        [100.0, 200.0],
        [ 95.0, 190.0],
        [ 90.0, 180.0],
        [ 85.0, 170.0],
        [ 80.0, 160.0],
        [ 75.0, 150.0],
        [ 70.0, 140.0],
    ])

    success_rate, exec_cost, avg_energy = compute_reach_avoid_success_rate(g_seq, h_seq, energy_seq)
    assert success_rate == 1.0
    assert exec_cost == 3.0
    # env0: first_success=2 -> energy[3]=85, 消耗15
    # env1: first_success=4 -> energy[5]=150, 消耗50
    assert abs(avg_energy - 32.5) < 1e-4, f"avg_energy: {avg_energy}"
    print("[PASS] test_success_rate_energy_alignment_uses_post_step_state")


# ---- Test 7: 训练脚本可导入 ----
def test_import_train_script():
    from legged_gym.scripts.train_ecfppo import train_ecfppo, HierarchicalVecEnv, create_env
    assert callable(train_ecfppo)
    assert callable(create_env)
    print("[PASS] test_import_train_script")


# ---- Test 8: num_observations 包含 energy 状态 ----
def test_num_obs_includes_energy():
    cfg = GO2HighLevelCfg()
    # num_observations 应该包含 +1 for energy state
    base = 8 + cfg.target_lidar_num_bins + cfg.lidar_num_bins + 1
    assert cfg.env.num_observations == base, \
        f"num_obs={cfg.env.num_observations} != expected {base}"
    print(f"  num_observations={cfg.env.num_observations}")
    print("[PASS] test_num_obs_includes_energy")


# ---- Test 9: buffer 正确存储 h_values ----
def test_buffer_stores_h_values():
    from rsl_rl.algorithms.ecfppo import EC_EFPPO_Buffer
    num_envs, horizon, obs_dim, act_dim = 4, 8, 10, 3
    buf = EC_EFPPO_Buffer(num_envs, horizon, (obs_dim,), (act_dim,), torch.device('cpu'))

    for step in range(horizon):
        buf.add(
            obs=torch.randn(num_envs, obs_dim),
            actions=torch.randn(num_envs, act_dim),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            value_reach=torch.randn(num_envs),
            energy=torch.rand(num_envs) * 400,
            energy_consumption=torch.rand(num_envs) * 5,
            g_values=torch.randn(num_envs),
            h_values=torch.randn(num_envs),
            dones=torch.zeros(num_envs),
            next_obs=torch.randn(num_envs, obs_dim),
            next_energy=torch.rand(num_envs) * 400,
            next_g=torch.randn(num_envs),
            next_h=torch.randn(num_envs),
        )

    assert buf.h_values.shape == (horizon + 1, num_envs)
    assert buf.step == horizon
    print("[PASS] test_buffer_stores_h_values")


# ---- Test 10: success rate 正确使用 h_values ----
def test_success_rate_uses_h_values():
    from legged_gym.scripts.train_ecfppo import compute_reach_avoid_success_rate

    T, N = 10, 4
    # g: 全部初始未到达，在 t=5 时到达目标
    g_seq = torch.ones(T, N) * 0.5
    g_seq[5:, :] = -0.5  # 全部在 t=5 到达
    # h: 前 2 个环境在 t=3 进入不安全区域（h >= 0），后 2 个始终安全
    h_seq = torch.ones(T, N) * -1.0
    h_seq[3:, :2] = 0.5  # env 0,1 在 t=3 不安全（早于 t=5 的到达）

    success_rate, _, _ = compute_reach_avoid_success_rate(g_seq, h_seq)
    # env 0,1: 到达前有 h >= 0 → 不安全 → 失败
    # env 2,3: 到达前 h < 0 → 安全 → 成功
    assert success_rate == 0.5, f"expected 0.5, got {success_rate}"
    print("[PASS] test_success_rate_uses_h_values")


# ---- Test 11: 端到端 mini 训练循环（mock） ----
def test_mini_training_loop():
    """用随机数据模拟一个完整的 mini 训练循环。"""
    torch.manual_seed(42)
    num_envs = 4
    horizon = 8
    obs_dim = 20
    act_dim = 3

    model = EC_EFPPO_ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=act_dim,
        hidden_dim=32, num_hidden_layers=2,
    )
    alg = EC_EFPPO(
        actor_critic=model,
        learning_rate=3e-4,
        gamma_energy=1.0,
        gamma_reach_init=0.999,
        gamma_reach_final=0.99999,
        gae_lambda=0.95,
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device='cpu',
    )
    alg.init_storage(num_envs, horizon, (obs_dim,), (act_dim,))

    obs = torch.randn(num_envs, obs_dim)
    g_vals = torch.randn(num_envs) * 3
    h_vals = torch.ones(num_envs) * -1.0  # 初始安全
    energy = torch.rand(num_envs) * 400

    # Rollout
    for step in range(horizon):
        actions, log_probs, vals_e, vals_r = alg.act(obs)
        next_obs = torch.randn(num_envs, obs_dim)
        next_g = torch.randn(num_envs) * 3
        next_h = torch.ones(num_envs) * -1.0
        next_energy = torch.rand(num_envs) * 400
        energy_consumption = torch.rand(num_envs) * 5
        dones = torch.zeros(num_envs)

        alg.buffer.add(
            obs=obs, actions=actions, log_probs=log_probs,
            values=vals_e, value_reach=vals_r,
            energy=energy, energy_consumption=energy_consumption,
            g_values=g_vals, h_values=h_vals, dones=dones,
            next_obs=next_obs, next_energy=next_energy, next_g=next_g, next_h=next_h,
        )
        obs = next_obs
        g_vals = next_g
        h_vals = next_h
        energy = next_energy

    # Bootstrap
    with torch.no_grad():
        last_e, last_r = model.evaluate(obs)

    # 计算优势
    gamma_reach = EC_EFPPO.compute_gamma_reach(0.999, 0.99999, 0, 100)
    alg.buffer.compute_advantages(last_e, last_r, 1.0, gamma_reach, 0.95, 0.999)

    # 更新
    loss_dict = alg.update(gamma_reach=gamma_reach, entropy_coef=0.01)

    assert 'actor_loss' in loss_dict
    assert 'energy_loss' in loss_dict
    assert 'reach_loss' in loss_dict
    assert all(np.isfinite(v) for v in loss_dict.values())
    print(f"  Losses: actor={loss_dict['actor_loss']:.4f}, "
          f"energy={loss_dict['energy_loss']:.4f}, "
          f"reach={loss_dict['reach_loss']:.4f}")
    print("[PASS] test_mini_training_loop")


if __name__ == '__main__':
    tests = [
        test_config_class,
        test_config_inheritance,
        test_model_with_config,
        test_alg_with_config,
        test_success_rate_with_energy,
        test_success_rate_without_energy,
        test_success_rate_energy_alignment_uses_post_step_state,
        test_import_train_script,
        test_num_obs_includes_energy,
        test_buffer_stores_h_values,
        test_success_rate_uses_h_values,
        test_mini_training_loop,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
