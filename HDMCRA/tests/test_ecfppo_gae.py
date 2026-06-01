"""
回归测试: Step 4 - EC-EFPPO GAE 算法
运行方式: conda run -n hdmcr python tests/test_ecfppo_gae.py
"""
import torch
import sys
sys.path.insert(0, '/home/caohy/repositories/HDMCRA/HDMCRA/rsl_rl')

from rsl_rl.algorithms.ecfppo_gae import calculate_indexs3, calculate_energy_gae, calculate_reach_gae

def make_test_data(T=10, nh=4, seed=42):
    torch.manual_seed(seed)
    device = 'cpu'
    dtype = torch.float32
    reward = torch.randn(T, nh, device=device, dtype=dtype) * 0.1
    energy = torch.randn(T + 1, nh, device=device, dtype=dtype) * 10
    h_seq = torch.randn(T + 1, nh, device=device, dtype=dtype) * 5
    values = torch.randn(T, nh, device=device, dtype=dtype)
    Vhs_seq = torch.randn(T + 1, nh, device=device, dtype=dtype)
    last_value = torch.randn(nh, device=device, dtype=dtype)
    last_value_reach = torch.randn(nh, device=device, dtype=dtype)
    return reward, energy, h_seq, values, Vhs_seq, last_value, last_value_reach

def test_indexs3_basic():
    reward, energy, h_seq, _, _, lv, lvr = make_test_data()
    indexs, done = calculate_indexs3(0.99, reward, energy, h_seq, lv, lvr)
    T, nh = reward.shape
    assert indexs.shape == (T, nh)
    assert done.shape == (T + 1, nh)
    assert torch.all((done == 0) | (done == 1))
    assert torch.all(done.sum(dim=0) >= 1)
    assert torch.all(indexs >= 0) and torch.all(indexs < T + 1)

def test_energy_gae_basic():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    done[3, 0] = 1.0
    adv, targets = calculate_energy_gae(0.99, 0.95, reward, values, done, lv)
    T, nh = reward.shape
    assert adv.shape == (T, nh)
    assert targets.shape == (T, nh)
    assert torch.allclose(targets, adv + values, atol=1e-5)
    assert torch.all(torch.isfinite(adv))

def test_reach_gae_basic():
    _, _, h_seq, _, Vhs_seq, _, _ = make_test_data()
    done = torch.zeros_like(h_seq)
    adv, targets = calculate_reach_gae(0.99, 0.95, h_seq, Vhs_seq, done)
    T = h_seq.shape[0] - 1
    assert adv.shape == (T, h_seq.shape[1])
    assert targets.shape == (T, h_seq.shape[1])
    assert torch.all(torch.isfinite(adv))

def test_edge_all_done():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.ones_like(reward)
    adv, _ = calculate_energy_gae(0.99, 0.95, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

def test_edge_no_done():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    adv, _ = calculate_energy_gae(0.99, 0.95, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

def test_edge_first_done():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    done[0] = 1.0
    adv, _ = calculate_energy_gae(0.99, 0.95, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

def test_edge_last_done():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    done[-1] = 1.0
    adv, _ = calculate_energy_gae(0.99, 0.95, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

def test_gamma_one():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    done[3, 0] = 1.0
    adv, _ = calculate_energy_gae(1.0, 0.95, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

def test_lam_zero():
    reward, _, _, values, _, lv, _ = make_test_data()
    done = torch.zeros_like(reward)
    done[3, 0] = 1.0
    adv, _ = calculate_energy_gae(0.99, 0.0, reward, values, done, lv)
    assert torch.all(torch.isfinite(adv))

if __name__ == '__main__':
    tests = [
        test_indexs3_basic, test_energy_gae_basic, test_reach_gae_basic,
        test_edge_all_done, test_edge_no_done, test_edge_first_done, test_edge_last_done,
        test_gamma_one, test_lam_zero
    ]
    for t in tests:
        t()
        print(f"✅ {t.__name__}")
    print(f"\n🎉 所有 {len(tests)} 个回归测试通过！")
