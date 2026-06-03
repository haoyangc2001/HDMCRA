"""
回归测试: Step 6 - EC-EFPPO 算法核心（Buffer + Trainer）
运行方式: conda run -n hdmcr python tests/test_ecfppo.py
"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA/rsl_rl')

from rsl_rl.algorithms.ecfppo import EC_EFPPO_Buffer, EC_EFPPO, EC_EFPPO_Batch
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


def make_model_and_buffer(num_envs=8, horizon=16, obs_dim=48, act_dim=12, seed=42):
    torch.manual_seed(seed)
    device = torch.device('cpu')
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=act_dim,
        hidden_dim=64, num_hidden_layers=2,  # 小网络加速测试
    )
    buffer = EC_EFPPO_Buffer(
        num_envs=num_envs, horizon=horizon,
        obs_shape=(obs_dim,), action_shape=(act_dim,), device=device,
    )
    return model, buffer, device


def fill_buffer_randomly(buffer, obs_dim=48, act_dim=12, seed=123):
    """用随机数据填充缓冲区。"""
    torch.manual_seed(seed)
    N = buffer.num_envs
    T = buffer.horizon
    for t in range(T):
        buffer.add(
            obs=torch.randn(N, obs_dim),
            actions=torch.randn(N, act_dim),
            log_probs=torch.randn(N),
            values=torch.randn(N),
            value_reach=torch.randn(N),
            energy=torch.rand(N) * 400,
            energy_consumption=torch.rand(N) * 5,
            g_values=torch.randn(N) * 3,
            h_values=torch.ones(N) * -1.0,
            dones=torch.zeros(N),
            next_obs=torch.randn(N, obs_dim),
            next_energy=torch.rand(N) * 400,
            next_g=torch.randn(N) * 3,
            next_h=torch.ones(N) * -1.0,
        )


# ---- Test 1: Buffer 实例化 ----
def test_buffer_instantiation():
    _, buffer, _ = make_model_and_buffer()
    assert buffer.num_envs == 8
    assert buffer.horizon == 16
    assert buffer.observations.shape == (17, 8, 48)  # T+1
    assert buffer.actions.shape == (16, 8, 12)
    assert buffer.values.shape == (16, 8)
    assert buffer.value_reach.shape == (16, 8)
    assert buffer.energy.shape == (17, 8)  # T+1
    assert buffer.energy_consumption.shape == (16, 8)
    assert buffer.g_values.shape == (17, 8)  # T+1
    assert buffer.h_values.shape == (17, 8)  # T+1
    assert buffer.dones.shape == (16, 8)
    assert buffer.step == 0
    print("[PASS] test_buffer_instantiation")


# ---- Test 2: Buffer add ----
def test_buffer_add():
    _, buffer, _ = make_model_and_buffer()
    N = buffer.num_envs
    for t in range(3):
        buffer.add(
            obs=torch.randn(N, 48),
            actions=torch.randn(N, 12),
            log_probs=torch.randn(N),
            values=torch.randn(N),
            value_reach=torch.randn(N),
            energy=torch.rand(N) * 400,
            energy_consumption=torch.rand(N) * 5,
            g_values=torch.randn(N) * 3,
            h_values=torch.ones(N) * -1.0,
            dones=torch.zeros(N),
            next_obs=torch.randn(N, 48),
            next_energy=torch.rand(N) * 400,
            next_g=torch.randn(N) * 3,
            next_h=torch.ones(N) * -1.0,
        )
    assert buffer.step == 3
    # 验证数据已存储（非零）
    assert buffer.observations[0].abs().sum() > 0
    print("[PASS] test_buffer_add")


# ---- Test 3: Buffer clear ----
def test_buffer_clear():
    _, buffer, _ = make_model_and_buffer()
    fill_buffer_randomly(buffer)
    assert buffer.step == buffer.horizon
    buffer.clear()
    assert buffer.step == 0
    print("[PASS] test_buffer_clear")


# ---- Test 4: Buffer compute_advantages 形状 ----
def test_compute_advantages_shapes():
    _, buffer, _ = make_model_and_buffer()
    fill_buffer_randomly(buffer)

    last_energy = torch.rand(8) * 400
    last_reach = torch.randn(8) * 3

    buffer.compute_advantages(
        last_energy, last_reach,
        gamma_energy=1.0, gamma_reach=0.999,
        gae_lambda=0.95, gamma_reach_init=0.999,
    )

    T = buffer.horizon
    N = buffer.num_envs
    assert buffer.advantages_total.shape == (T, N)
    assert buffer.targets_energy.shape == (T, N)
    assert buffer.targets_reach.shape == (T, N)
    # 值应为有限数
    assert torch.isfinite(buffer.advantages_total).all()
    assert torch.isfinite(buffer.targets_energy).all()
    assert torch.isfinite(buffer.targets_reach).all()
    print("[PASS] test_compute_advantages_shapes")


# ---- Test 5: Buffer iter_batches ----
def test_iter_batches():
    _, buffer, _ = make_model_and_buffer()
    fill_buffer_randomly(buffer)

    last_energy = torch.rand(8) * 400
    last_reach = torch.randn(8) * 3
    buffer.compute_advantages(last_energy, last_reach, 1.0, 0.999, 0.95, 0.999)

    batches = list(buffer.iter_batches(num_mini_batches=2, num_epochs=2))
    total_samples = 0
    for batch in batches:
        assert isinstance(batch, EC_EFPPO_Batch)
        assert batch.observations.dim() == 2
        assert batch.actions.dim() == 2
        total_samples += batch.observations.size(0)

    # 总样本数 = horizon * num_envs * num_epochs
    expected = buffer.horizon * buffer.num_envs * 2  # 2 epochs
    assert total_samples == expected, f"total {total_samples} != expected {expected}"
    print("[PASS] test_iter_batches")


# ---- Test 6: EC_EFPPO 实例化（三个优化器） ----
def test_ecefppo_instantiation():
    model, _, device = make_model_and_buffer()
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
    # 三个优化器应存在
    assert alg.policy_optimizer is not None
    assert alg.energy_optimizer is not None
    assert alg.reach_optimizer is not None
    # 优化器的参数组应分别对应三个子网络
    policy_params = set(id(p) for p in alg.policy_optimizer.param_groups[0]['params'])
    energy_params = set(id(p) for p in alg.energy_optimizer.param_groups[0]['params'])
    reach_params = set(id(p) for p in alg.reach_optimizer.param_groups[0]['params'])
    assert policy_params.isdisjoint(energy_params)
    assert policy_params.isdisjoint(reach_params)
    assert energy_params.isdisjoint(reach_params)
    print("[PASS] test_ecefppo_instantiation")


# ---- Test 7: EC_EFPPO act ----
def test_ecefppo_act():
    model, _, device = make_model_and_buffer()
    alg = EC_EFPPO(actor_critic=model, device='cpu')
    obs = torch.randn(8, 48)
    actions, log_probs, vals_e, vals_r = alg.act(obs)
    assert actions.shape == (8, 12)
    assert log_probs.shape == (8,)
    assert vals_e.shape == (8,)
    assert vals_r.shape == (8,)
    print("[PASS] test_ecefppo_act")


# ---- Test 8: EC_EFPPO update 完整流程 ----
def test_ecefppo_update():
    model, buffer, device = make_model_and_buffer(num_envs=8, horizon=16, obs_dim=48, act_dim=12)
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
    alg.init_storage(num_envs=8, horizon=16, obs_shape=(48,), action_shape=(12,))

    # 用模型收集数据
    for t in range(16):
        obs = torch.randn(8, 48)
        with torch.no_grad():
            actions, log_probs, vals_e, vals_r = model.act(obs)
        buffer = alg.buffer
        buffer.add(
            obs=obs, actions=actions, log_probs=log_probs,
            values=vals_e, value_reach=vals_r,
            energy=torch.rand(8) * 400,
            energy_consumption=torch.rand(8) * 5,
            g_values=torch.randn(8) * 3,
            h_values=torch.ones(8) * -1.0,
            dones=torch.zeros(8),
            next_obs=torch.randn(8, 48),
            next_energy=torch.rand(8) * 400,
            next_g=torch.randn(8) * 3,
            next_h=torch.ones(8) * -1.0,
        )

    # 计算优势
    alg.buffer.compute_advantages(
        torch.rand(8) * 400, torch.randn(8) * 3,
        1.0, 0.999, 0.95, 0.999,
    )

    # 更新
    loss_dict = alg.update(gamma_reach=0.999, entropy_coef=0.01)

    assert 'actor_loss' in loss_dict
    assert 'energy_loss' in loss_dict
    assert 'reach_loss' in loss_dict
    assert 'entropy_loss' in loss_dict
    assert all(np.isfinite(v) for v in loss_dict.values())
    print(f"  Losses: actor={loss_dict['actor_loss']:.4f}, "
          f"energy={loss_dict['energy_loss']:.4f}, "
          f"reach={loss_dict['reach_loss']:.4f}, "
          f"entropy={loss_dict['entropy_loss']:.4f}")
    print("[PASS] test_ecefppo_update")


# ---- Test 9: 三路梯度独立性 ----
def test_independent_gradient_flow():
    """验证 update 后三个子网络的参数确实发生了变化。"""
    torch.manual_seed(42)
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=48, num_critic_obs=48, num_actions=12,
        hidden_dim=64, num_hidden_layers=2,
    )
    alg = EC_EFPPO(
        actor_critic=model, learning_rate=1e-2,
        num_learning_epochs=4, num_mini_batches=4,
        clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.05,
        device='cpu',
    )
    alg.init_storage(num_envs=16, horizon=32, obs_shape=(48,), action_shape=(12,))

    # 记录初始参数
    actor_init = {n: p.clone() for n, p in model.actor.named_parameters()}
    energy_init = {n: p.clone() for n, p in model.energy_critic.named_parameters()}
    reach_init = {n: p.clone() for n, p in model.reach_critic.named_parameters()}

    # 收集数据
    for t in range(32):
        obs = torch.randn(16, 48)
        with torch.no_grad():
            a, lp, ve, vr = model.act(obs)
        alg.buffer.add(
            obs=obs, actions=a, log_probs=lp,
            values=ve, value_reach=vr,
            energy=torch.rand(16) * 400,
            energy_consumption=torch.rand(16) * 5,
            g_values=torch.randn(16) * 3,
            h_values=torch.ones(16) * -1.0,
            dones=torch.zeros(16),
            next_obs=torch.randn(16, 48),
            next_energy=torch.rand(16) * 400,
            next_g=torch.randn(16) * 3,
            next_h=torch.ones(16) * -1.0,
        )

    alg.buffer.compute_advantages(
        torch.rand(16) * 400, torch.randn(16) * 3,
        1.0, 0.999, 0.95, 0.999,
    )
    alg.update(gamma_reach=0.999, entropy_coef=0.05)

    # 检查参数变化
    actor_changed = any(
        not torch.equal(p, actor_init[n])
        for n, p in model.actor.named_parameters()
    )
    energy_changed = any(
        not torch.equal(p, energy_init[n])
        for n, p in model.energy_critic.named_parameters()
    )
    reach_changed = any(
        not torch.equal(p, reach_init[n])
        for n, p in model.reach_critic.named_parameters()
    )
    assert actor_changed, "actor 参数未变化"
    assert energy_changed, "energy_critic 参数未变化"
    assert reach_changed, "reach_critic 参数未变化"
    print("[PASS] test_independent_gradient_flow")


# ---- Test 10: gamma_reach 退火 ----
def test_gamma_reach_annealing():
    gamma = EC_EFPPO.compute_gamma_reach(0.999, 0.99999, 0, 100)
    assert abs(gamma - 0.999) < 1e-6, f"初始 gamma: {gamma}"

    gamma = EC_EFPPO.compute_gamma_reach(0.999, 0.99999, 50, 100)
    expected = min(0.99999, 0.999 + (0.99999 - 0.999) * 50 * 2 / 100)
    assert abs(gamma - expected) < 1e-6, f"中间 gamma: {gamma} != {expected}"

    gamma = EC_EFPPO.compute_gamma_reach(0.999, 0.99999, 100, 100)
    assert abs(gamma - 0.99999) < 1e-6, f"最终 gamma: {gamma}"
    print("[PASS] test_gamma_reach_annealing")


# ---- Test 11: entropy 退火 ----
def test_entropy_annealing():
    coef = EC_EFPPO.compute_entropy_coef(0.01, 0, 100, anneal=True)
    assert abs(coef - 0.01) < 1e-6, f"初始 entropy: {coef}"

    coef = EC_EFPPO.compute_entropy_coef(0.01, 50, 100, anneal=True)
    expected = 0.01 * (100 - 50) / 100
    assert abs(coef - expected) < 1e-6, f"中间 entropy: {coef}"

    coef = EC_EFPPO.compute_entropy_coef(0.01, 100, 100, anneal=True)
    assert abs(coef) < 1e-6, f"最终 entropy: {coef}"

    coef = EC_EFPPO.compute_entropy_coef(0.01, 50, 100, anneal=False)
    assert abs(coef - 0.01) < 1e-6, f"关闭退火: {coef}"
    print("[PASS] test_entropy_annealing")


# ---- Test 12: init_storage ----
def test_init_storage():
    model, _, _ = make_model_and_buffer()
    alg = EC_EFPPO(actor_critic=model, device='cpu')
    assert alg.buffer is None
    alg.init_storage(num_envs=8, horizon=16, obs_shape=(48,), action_shape=(12,))
    assert alg.buffer is not None
    assert isinstance(alg.buffer, EC_EFPPO_Buffer)
    print("[PASS] test_init_storage")


# ---- Test 13: Buffer compute_advantages 未完成 rollout 报错 ----
def test_incomplete_rollout_error():
    _, buffer, _ = make_model_and_buffer()
    fill_buffer_randomly(buffer, seed=99)
    # 只填充一半
    buffer.clear()
    for t in range(8):
        buffer.add(
            obs=torch.randn(8, 48), actions=torch.randn(8, 12),
            log_probs=torch.randn(8), values=torch.randn(8),
            value_reach=torch.randn(8), energy=torch.rand(8) * 400,
            energy_consumption=torch.rand(8) * 5, g_values=torch.randn(8) * 3,
            h_values=torch.ones(8) * -1.0, dones=torch.zeros(8),
            next_obs=torch.randn(8, 48), next_energy=torch.rand(8) * 400,
            next_g=torch.randn(8) * 3, next_h=torch.ones(8) * -1.0,
        )
    try:
        buffer.compute_advantages(torch.zeros(8), torch.zeros(8), 1.0, 0.999, 0.95, 0.999)
        assert False, "应抛出 RuntimeError"
    except RuntimeError as e:
        assert "incomplete" in str(e).lower()
    print("[PASS] test_incomplete_rollout_error")


if __name__ == '__main__':
    tests = [
        test_buffer_instantiation,
        test_buffer_add,
        test_buffer_clear,
        test_compute_advantages_shapes,
        test_iter_batches,
        test_ecefppo_instantiation,
        test_ecefppo_act,
        test_ecefppo_update,
        test_independent_gradient_flow,
        test_gamma_reach_annealing,
        test_entropy_annealing,
        test_init_storage,
        test_incomplete_rollout_error,
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
