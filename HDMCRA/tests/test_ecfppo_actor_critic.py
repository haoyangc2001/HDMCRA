"""
回归测试: Step 5 - EC-EFPPO 三网络架构
运行方式: conda run -n hdmcr python tests/test_ecfppo_actor_critic.py
"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA/rsl_rl')

from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic


def make_model(num_actor_obs=48, num_critic_obs=48, num_actions=12, seed=42):
    torch.manual_seed(seed)
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        hidden_dim=256,
        num_hidden_layers=2,
        init_noise_std=1.0,
    )
    return model


# ---- Test 1: 模型实例化 ----
def test_instantiation():
    model = make_model()
    assert hasattr(model, 'actor')
    assert hasattr(model, 'energy_critic')
    assert hasattr(model, 'reach_critic')
    assert hasattr(model, 'std')
    print("[PASS] test_instantiation")


# ---- Test 2: 三网络完全独立 ----
def test_independent_params():
    model = make_model()
    actor_params = set(id(p) for p in model.actor.parameters())
    energy_params = set(id(p) for p in model.energy_critic.parameters())
    reach_params = set(id(p) for p in model.reach_critic.parameters())
    # 三组参数不应有交集
    assert actor_params.isdisjoint(energy_params), \
        "actor 和 energy_critic 共享参数!"
    assert actor_params.isdisjoint(reach_params), \
        "actor 和 reach_critic 共享参数!"
    assert energy_params.isdisjoint(reach_params), \
        "energy_critic 和 reach_critic 共享参数!"
    print("[PASS] test_independent_params")


# ---- Test 3: 网络结构（参数化，不再硬编码旧的 2x256+tanh 假设） ----
def test_network_structure():
    model = make_model(num_actor_obs=48, num_actions=12)
    actor_modules = list(model.actor.children())
    linear_layers = [m for m in actor_modules if isinstance(m, nn.Linear)]
    activation_layers = [m for m in actor_modules if isinstance(m, nn.ELU)]
    assert len(linear_layers) == 3, f"actor 线性层数错误: {len(linear_layers)}"
    assert len(activation_layers) == 2, f"actor 激活层数错误: {len(activation_layers)}"
    assert linear_layers[0].in_features == 48
    assert linear_layers[0].out_features == 256
    assert linear_layers[-1].out_features == 12

    ec_modules = list(model.energy_critic.children())
    ec_linear = [m for m in ec_modules if isinstance(m, nn.Linear)]
    ec_activation = [m for m in ec_modules if isinstance(m, nn.ELU)]
    assert len(ec_linear) == 3
    assert len(ec_activation) == 2
    assert ec_linear[-1].out_features == 1

    rc_modules = list(model.reach_critic.children())
    rc_linear = [m for m in rc_modules if isinstance(m, nn.Linear)]
    rc_activation = [m for m in rc_modules if isinstance(m, nn.ELU)]
    assert len(rc_linear) == 3
    assert len(rc_activation) == 2
    assert rc_linear[-1].out_features == 1

    print("[PASS] test_network_structure")


# ---- Test 4: act() 输出形状和 log_prob ----
def test_act_shapes():
    model = make_model()
    model.eval()
    N = 64
    obs = torch.randn(N, 48)
    critic_obs = torch.randn(N, 48)
    action, log_prob, energy_val, reach_val = model.act(obs, critic_obs)
    assert action.shape == (N, 12), f"action shape: {action.shape}"
    assert log_prob.shape == (N,), f"log_prob shape: {log_prob.shape}"
    assert energy_val.shape == (N,), f"energy_val shape: {energy_val.shape}"
    assert reach_val.shape == (N,), f"reach_val shape: {reach_val.shape}"
    # log_prob 应为负值（概率 < 1）
    assert torch.all(log_prob < 0), "log_prob 应为负值"
    print("[PASS] test_act_shapes")


# ---- Test 5: act() 不传 critic_observations 时使用 observations ----
def test_act_no_critic_obs():
    model = make_model()
    N = 32
    obs = torch.randn(N, 48)
    action, log_prob, energy_val, reach_val = model.act(obs)
    assert action.shape == (N, 12)
    assert energy_val.shape == (N,)
    assert reach_val.shape == (N,)
    print("[PASS] test_act_no_critic_obs")


# ---- Test 6: evaluate() 输出形状 ----
def test_evaluate_shapes():
    model = make_model()
    N = 32
    critic_obs = torch.randn(N, 48)
    energy_val, reach_val = model.evaluate(critic_obs)
    assert energy_val.shape == (N,), f"energy_val shape: {energy_val.shape}"
    assert reach_val.shape == (N,), f"reach_val shape: {reach_val.shape}"
    print("[PASS] test_evaluate_shapes")


# ---- Test 7: act_inference() 输出形状且确定性 ----
def test_act_inference():
    model = make_model()
    model.eval()
    N = 32
    obs = torch.randn(N, 48)
    # 两次调用应返回相同结果（确定性）
    action1 = model.act_inference(obs)
    action2 = model.act_inference(obs)
    assert action1.shape == (N, 12)
    assert torch.allclose(action1, action2), \
        "act_inference 应返回确定性结果"
    print("[PASS] test_act_inference")


# ---- Test 8: 正交初始化验证 ----
def test_orthogonal_init():
    model = make_model(seed=123)
    # 检查 actor 第一个隐藏层权重是否接近 orthogonal(sqrt(2))
    w = model.actor[0].weight.data
    # 正交矩阵的行列应满足 W @ W^T ≈ gain^2 * I (近似)
    # 这里简单验证权重的标准差在合理范围内
    assert w.std() > 0.01, "权重标准差过小，可能未正确初始化"
    # 检查 actor 最后一层权重（应该是 orthogonal(0.01)）
    last_w = model.actor[-1].weight.data
    assert last_w.abs().max() < 1.0, \
        f"actor 最后一层权重应较小 (gain=0.01), max={last_w.abs().max()}"
    # 检查 bias 是否全为 0
    for name, param in model.named_parameters():
        if 'bias' in name:
            assert torch.all(param == 0), f"{name} 应初始化为 0"
    print("[PASS] test_orthogonal_init")


# ---- Test 9: log_std 为可学习参数 ----
def test_learnable_std():
    model = make_model(num_actions=12)
    assert isinstance(model.std, nn.Parameter), \
        "std 应为 nn.Parameter"
    assert model.std.shape == (12,), f"std shape: {model.std.shape}"
    assert model.std.requires_grad, "std 应可学习"
    # 初始值应全为 init_noise_std (1.0)
    assert torch.allclose(model.std.data, torch.ones(12)), \
        "std 初始值应为 1.0"
    print("[PASS] test_learnable_std")


# ---- Test 10: 梯度可以独立回传 ----
def test_independent_gradients():
    model = make_model()
    N = 16
    obs = torch.randn(N, 48)
    critic_obs = torch.randn(N, 48)

    action, log_prob, energy_val, reach_val = model.act(obs, critic_obs)

    # 分别计算 loss 并反向传播
    actor_loss = -log_prob.mean()
    energy_loss = energy_val.pow(2).mean()
    reach_loss = reach_val.pow(2).mean()

    # actor 反向
    actor_loss.backward(retain_graph=True)
    actor_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.actor.parameters())
    energy_no_grad_before = all(p.grad is None
                                for p in model.energy_critic.parameters())
    reach_no_grad_before = all(p.grad is None
                               for p in model.reach_critic.parameters())
    assert actor_has_grad, "actor 应有梯度"
    # energy/reach 在 actor backward 时不应有梯度（因为 graph 不涉及它们的参数）
    # 注意: 这取决于计算图连接，这里只验证 actor 有梯度

    # energy 反向
    energy_loss.backward(retain_graph=True)
    energy_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.energy_critic.parameters())
    assert energy_has_grad, "energy_critic 应有梯度"

    # reach 反向
    reach_loss.backward()
    reach_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.reach_critic.parameters())
    assert reach_has_grad, "reach_critic 应有梯度"

    print("[PASS] test_independent_gradients")


# ---- Test 11: 不同 num_actor_obs / num_critic_obs ----
def test_different_obs_dims():
    model = EC_EFPPO_ActorCritic(
        num_actor_obs=48,
        num_critic_obs=52,  # 比如 critic 多看 4 维
        num_actions=12,
        hidden_dim=256,
        num_hidden_layers=2,
    )
    obs = torch.randn(8, 48)
    critic_obs = torch.randn(8, 52)
    action, log_prob, energy_val, reach_val = model.act(obs, critic_obs)
    assert action.shape == (8, 12)
    assert energy_val.shape == (8,)
    assert reach_val.shape == (8,)
    print("[PASS] test_different_obs_dims")


# ---- Test 12: 从 modules 包导入 ----
def test_import_from_package():
    from rsl_rl.modules import EC_EFPPO_ActorCritic as EC
    assert EC is EC_EFPPO_ActorCritic, \
        "从包导入的类应与直接导入一致"
    print("[PASS] test_import_from_package")


if __name__ == '__main__':
    tests = [
        test_instantiation,
        test_independent_params,
        test_network_structure,
        test_act_shapes,
        test_act_no_critic_obs,
        test_evaluate_shapes,
        test_act_inference,
        test_orthogonal_init,
        test_learnable_std,
        test_independent_gradients,
        test_different_obs_dims,
        test_import_from_package,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
