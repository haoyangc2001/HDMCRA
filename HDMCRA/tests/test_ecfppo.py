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


def test_buffer_time_semantics():
    """P1: buffer 的 [t] / [t+1] 语义必须稳定。"""
    _, buffer, _ = make_model_and_buffer(num_envs=2, horizon=2, obs_dim=3, act_dim=1)
    obs0 = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    next_obs0 = torch.tensor([[1.5, 0.0, 0.0], [2.5, 0.0, 0.0]])
    obs1 = next_obs0.clone()
    next_obs1 = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    buffer.add(
        obs=obs0, actions=torch.zeros(2, 1), log_probs=torch.zeros(2),
        values=torch.zeros(2), value_reach=torch.zeros(2),
        energy=torch.tensor([10.0, 20.0]), energy_consumption=torch.tensor([1.0, 2.0]),
        g_values=torch.tensor([5.0, 6.0]), h_values=torch.tensor([-1.0, -1.0]),
        dones=torch.zeros(2), next_obs=next_obs0, next_energy=torch.tensor([9.0, 18.0]),
        next_g=torch.tensor([4.0, 5.0]), next_h=torch.tensor([-1.0, -1.0]),
    )
    buffer.add(
        obs=obs1, actions=torch.zeros(2, 1), log_probs=torch.zeros(2),
        values=torch.zeros(2), value_reach=torch.zeros(2),
        energy=torch.tensor([9.0, 18.0]), energy_consumption=torch.tensor([1.0, 3.0]),
        g_values=torch.tensor([4.0, 5.0]), h_values=torch.tensor([-1.0, -1.0]),
        dones=torch.zeros(2), next_obs=next_obs1, next_energy=torch.tensor([8.0, 15.0]),
        next_g=torch.tensor([3.0, 4.0]), next_h=torch.tensor([-1.0, -1.0]),
    )

    assert torch.allclose(buffer.observations[0], obs0)
    assert torch.allclose(buffer.observations[1], next_obs0)
    assert torch.allclose(buffer.observations[2], next_obs1)
    assert torch.allclose(buffer.energy[0], torch.tensor([10.0, 20.0]))
    assert torch.allclose(buffer.energy[1], torch.tensor([9.0, 18.0]))
    assert torch.allclose(buffer.energy[2], torch.tensor([8.0, 15.0]))
    assert torch.allclose(buffer.g_values[1], torch.tensor([4.0, 5.0]))
    print("[PASS] test_buffer_time_semantics")


def test_combined_advantage_uses_gamma_reach_init():
    """P1: combined advantage 当前约定使用 gamma_reach_init，而不是 gamma_reach。"""
    _, buffer, _ = make_model_and_buffer(num_envs=2, horizon=4, obs_dim=6, act_dim=2)
    fill_buffer_randomly(buffer, obs_dim=6, act_dim=2, seed=321)
    last_energy = torch.rand(2) * 10
    last_reach = torch.randn(2)

    buffer.compute_advantages(
        last_energy, last_reach,
        gamma_energy=0.99, gamma_reach=0.5,
        gae_lambda=0.95, gamma_reach_init=0.999,
    )
    adv_with_init = buffer.advantages_total.clone()

    buffer.compute_advantages(
        last_energy, last_reach,
        gamma_energy=0.99, gamma_reach=0.1,
        gae_lambda=0.95, gamma_reach_init=0.999,
    )
    adv_with_same_init = buffer.advantages_total.clone()

    assert torch.allclose(adv_with_init, adv_with_same_init), \
        'combined advantage should depend on gamma_reach_init, not current gamma_reach'
    print("[PASS] test_combined_advantage_uses_gamma_reach_init")


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


def test_energy_action_debug_stats():
    """D004: energy/action debug 字段只反映 rollout 统计，不改变训练逻辑。"""
    device = torch.device('cpu')
    num_envs = 2
    horizon = 3
    obs_dim = 3
    act_dim = 2
    buffer = EC_EFPPO_Buffer(
        num_envs=num_envs, horizon=horizon, obs_shape=(obs_dim,),
        action_shape=(act_dim,), device=device,
    )

    energies = [
        torch.tensor([10.0, 20.0]),
        torch.tensor([5.0, 10.0]),
        torch.tensor([-400.0, 0.0]),
        torch.tensor([-400.0, -1.0]),
    ]
    actions = [
        torch.tensor([[0.0, 1.0], [1.2, -0.5]]),
        torch.tensor([[-1.0, 0.2], [0.1, 0.2]]),
        torch.tensor([[0.99, -1.01], [0.0, 0.0]]),
    ]
    consumptions = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
    ]
    action_means = [
        torch.tensor([[0.2, 0.8], [1.1, -0.4]]),
        torch.tensor([[-0.8, 0.3], [0.0, 0.1]]),
        torch.tensor([[0.4, -1.2], [0.0, 0.0]]),
    ]

    for t in range(horizon):
        buffer.add(
            obs=torch.zeros(num_envs, obs_dim),
            actions=actions[t],
            log_probs=torch.zeros(num_envs),
            values=torch.zeros(num_envs),
            value_reach=torch.zeros(num_envs),
            energy=energies[t],
            energy_consumption=consumptions[t],
            g_values=torch.ones(num_envs) * 500.0,
            h_values=torch.ones(num_envs) * -300.0,
            dones=torch.zeros(num_envs),
            next_obs=torch.zeros(num_envs, obs_dim),
            next_energy=energies[t + 1],
            next_g=torch.ones(num_envs) * 500.0,
            next_h=torch.ones(num_envs) * -300.0,
            action_mean=action_means[t],
        )

    buffer.compute_advantages(
        last_values_energy=torch.zeros(num_envs),
        last_values_reach=torch.zeros(num_envs),
        gamma_energy=0.99, gamma_reach=0.999,
        gae_lambda=0.95, gamma_reach_init=0.999,
    )

    stats = buffer.debug_stats
    assert abs(stats['energy_consumption_mean'] - 3.5) < 1e-6
    assert abs(stats['energy_consumption_max'] - 6.0) < 1e-6
    assert abs(stats['init_energy_min'] - 10.0) < 1e-6
    assert abs(stats['init_energy_mean'] - 15.0) < 1e-6
    assert abs(stats['init_energy_max'] - 20.0) < 1e-6
    assert abs(stats['first_energy_min_step_mean'] - 3.0) < 1e-6
    assert abs(stats['action_clip_ratio'] - (4.0 / 12.0)) < 1e-6
    assert abs(stats['action_abs_mean'] - (6.2 / 12.0)) < 1e-6
    assert abs(stats['clipped_action_abs_mean'] - (5.99 / 12.0)) < 1e-6
    assert abs(stats['clipped_action_abs_max'] - 1.0) < 1e-6
    assert abs(stats['action_mean_abs_mean'] - (5.3 / 12.0)) < 1e-6
    assert abs(stats['action_mean_abs_max'] - 1.2) < 1e-6
    assert abs(stats['action_mean_clip_ratio'] - (2.0 / 12.0)) < 1e-6
    assert abs(stats['action_mean_abs_mean_dim0'] - (2.5 / 6.0)) < 1e-6
    assert abs(stats['action_mean_abs_mean_dim1'] - (2.8 / 6.0)) < 1e-6
    assert abs(stats['action_mean_clip_ratio_dim0'] - (1.0 / 6.0)) < 1e-6
    assert abs(stats['action_mean_clip_ratio_dim1'] - (1.0 / 6.0)) < 1e-6
    assert abs(stats['clipped_action_abs_mean_dim0'] - (3.09 / 6.0)) < 1e-6
    assert abs(stats['clipped_action_abs_mean_dim1'] - (2.9 / 6.0)) < 1e-6
    print("[PASS] test_energy_action_debug_stats")


def test_reach_bootstrap_value_clip_bounds_targets():
    """P0: 极端 reach bootstrap 不应把 open 样本 target 拉到无语义量级。"""
    device = torch.device('cpu')
    num_envs = 2
    horizon = 4
    obs_dim = 3
    act_dim = 1
    buffer = EC_EFPPO_Buffer(
        num_envs=num_envs, horizon=horizon, obs_shape=(obs_dim,),
        action_shape=(act_dim,), device=device, reach_value_clip=5000.0,
    )

    for _ in range(horizon):
        buffer.add(
            obs=torch.zeros(num_envs, obs_dim),
            actions=torch.zeros(num_envs, act_dim),
            log_probs=torch.zeros(num_envs),
            values=torch.zeros(num_envs),
            value_reach=torch.zeros(num_envs),
            energy=torch.ones(num_envs) * -400.0,
            energy_consumption=torch.zeros(num_envs),
            g_values=torch.ones(num_envs) * 500.0,
            h_values=torch.ones(num_envs) * -300.0,
            dones=torch.zeros(num_envs),
            next_obs=torch.zeros(num_envs, obs_dim),
            next_energy=torch.ones(num_envs) * -400.0,
            next_g=torch.ones(num_envs) * 500.0,
            next_h=torch.ones(num_envs) * -300.0,
        )

    last_energy = torch.zeros(num_envs)
    last_reach = torch.ones(num_envs) * -1e9
    buffer.compute_advantages(
        last_energy, last_reach,
        gamma_energy=0.99, gamma_reach=0.999,
        gae_lambda=0.95, gamma_reach_init=0.999,
    )

    assert buffer.debug_stats['reach_value_clip_ratio'] > 0.0
    assert buffer.targets_reach.min().item() >= -5000.0 - 1e-3
    assert abs(buffer.debug_stats['targets_reach_min_next_value_reach']) <= 5000.0
    print("[PASS] test_reach_bootstrap_value_clip_bounds_targets")


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
    assert id(model.log_std) in policy_params, "policy optimizer 必须包含动作 log_std 参数"
    assert policy_params.isdisjoint(energy_params)
    assert policy_params.isdisjoint(reach_params)
    assert energy_params.isdisjoint(reach_params)
    print("[PASS] test_ecefppo_instantiation")


def test_policy_optimizer_updates_std():
    """P0: entropy/std 诊断要求 std 必须由 policy optimizer 实际更新。"""
    model, _, _ = make_model_and_buffer(num_envs=4, horizon=4, obs_dim=8, act_dim=3)
    alg = EC_EFPPO(
        actor_critic=model,
        learning_rate=1e-2,
        entropy_coef=0.01,
        device='cpu',
    )

    obs = torch.randn(16, 8)
    old_std = model.std.detach().clone()

    alg.policy_optimizer.zero_grad()
    model.update_distribution(obs)
    loss = -model.entropy.mean()
    loss.backward()
    alg.policy_optimizer.step()
    model.clamp_log_std_()

    assert not torch.allclose(model.std.detach(), old_std), "std 应在 policy optimizer step 后变化"
    assert torch.all(model.std.detach() > 0), "std 必须保持正值"
    assert torch.all(model.log_std.detach() <= model.log_std_max + 1e-6), "log_std 不应超过上界"
    assert torch.all(model.log_std.detach() >= model.log_std_min - 1e-6), "log_std 不应低于下界"
    print("[PASS] test_policy_optimizer_updates_std")


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
    assert 'mean_bound_loss' in loss_dict
    assert loss_dict['mean_bound_loss'] >= 0.0
    assert all(np.isfinite(v) for v in loss_dict.values())
    print(f"  Losses: actor={loss_dict['actor_loss']:.4f}, "
          f"energy={loss_dict['energy_loss']:.4f}, "
          f"reach={loss_dict['reach_loss']:.4f}, "
          f"entropy={loss_dict['entropy_loss']:.4f}")
    print("[PASS] test_ecefppo_update")


def test_actor_mean_bound_loss():
    """D007: actor mean 边界正则只惩罚越过动作执行边界的均值。"""
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=4, num_critic_obs=4, num_actions=3,
        hidden_dim=8, num_hidden_layers=2,
    )
    alg = EC_EFPPO(
        actor_critic=model, actor_mean_bound=1.0,
        actor_mean_bound_coef=1e-2, device='cpu',
    )

    inside = torch.tensor([[0.0, 0.5, -1.0]])
    outside = torch.tensor([[2.0, -3.0, 1.5]])

    assert alg._actor_mean_bound_loss(inside).item() == 0.0
    assert alg._actor_mean_bound_loss(outside).item() > 0.0
    assert alg.actor_mean_bound == 1.0
    assert alg.actor_mean_bound_coef == 1e-2
    print("[PASS] test_actor_mean_bound_loss")


def test_policy_gae_direction_for_cost_like_advantages():
    """D008: 更小的 reach-avoid advantage 应提高对应动作概率。"""
    advantages_total = torch.tensor([-1.0, 1.0])
    gae = EC_EFPPO._policy_gae_from_advantages(advantages_total)
    assert gae[0].item() > 0.0
    assert gae[1].item() < 0.0

    log_probs = torch.tensor([0.0, 0.0], requires_grad=True)
    old_log_probs = torch.zeros(2)
    ratio = torch.exp(log_probs - old_log_probs)
    loss_actor1 = ratio * gae
    loss_actor2 = torch.clamp(ratio, 0.8, 1.2) * gae
    policy_loss = -torch.min(loss_actor1, loss_actor2).mean()
    policy_loss.backward()

    # 梯度下降会增大低 cost-like advantage 样本的 log_prob，降低高样本的 log_prob。
    assert log_probs.grad[0].item() < 0.0
    assert log_probs.grad[1].item() > 0.0
    print("[PASS] test_policy_gae_direction_for_cost_like_advantages")


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
        test_buffer_time_semantics,
        test_combined_advantage_uses_gamma_reach_init,
        test_buffer_instantiation,
        test_buffer_add,
        test_buffer_clear,
        test_compute_advantages_shapes,
        test_energy_action_debug_stats,
        test_reach_bootstrap_value_clip_bounds_targets,
        test_iter_batches,
        test_ecefppo_instantiation,
        test_policy_optimizer_updates_std,
        test_ecefppo_act,
        test_ecefppo_update,
        test_actor_mean_bound_loss,
        test_policy_gae_direction_for_cost_like_advantages,
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
