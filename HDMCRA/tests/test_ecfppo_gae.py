"""
回归测试: Step 4 / Step 17 P0 - EC-EFPPO GAE 算法
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


def reference_calculate_indexs3(gamma, reward, energy, T_hs, last_value, last_value_reach):
    T, nh = reward.shape
    Vs_row = torch.full((T, nh), float('inf'), dtype=reward.dtype)
    Vhs_row = torch.full((T, nh), float('inf'), dtype=reward.dtype)
    done = torch.zeros(T + 1, nh, dtype=reward.dtype)
    mask_1 = torch.full((T, 1), float('inf'), dtype=reward.dtype)
    indexs = torch.zeros(T, nh, dtype=torch.long)
    for ii in range(T - 1, -1, -1):
        Vs_row = mask_1 * (reward[ii] + gamma * Vs_row)
        Vs_row[ii] = 0.0
        Vhs_row = Vhs_row.clone()
        Vhs_row[ii] = T_hs[ii]
        V_total = torch.maximum(Vs_row - energy[T - ii - 1], Vhs_row).flip(0)
        V_next = torch.maximum((gamma ** ii) * last_value + V_total[-1] - energy[T - ii - 1], last_value_reach)
        V_total_1 = torch.cat([V_total, V_next.unsqueeze(0)], dim=0)
        index = torch.argmin(V_total_1, dim=0)
        done[index, torch.arange(nh)] = 1.0
        indexs[ii] = index
        mask_1 = torch.roll(mask_1, shifts=1, dims=0)
        mask_1[0] = 1.0
    return indexs, done


def reference_calculate_reach_gae(gamma, gae_lambda, h_seq, Vhs_seq, done):
    Tp1, nh = h_seq.shape
    T = Tp1 - 1
    done = done.to(torch.long)
    lam_ratio = gae_lambda / max(1.0 - gae_lambda, 1e-6)
    gae_coeffs = torch.zeros(T + 1, nh, dtype=h_seq.dtype)
    value_table = torch.zeros(T + 1, nh, dtype=h_seq.dtype)
    value_table[0] = Vhs_seq[T]
    pre_done = torch.zeros(nh, dtype=h_seq.dtype)
    q_targets = torch.zeros(T, nh, dtype=h_seq.dtype)

    for ii in range(T - 1, -1, -1):
        done_row = done[ii].to(h_seq.dtype).unsqueeze(0)
        pre_done_row = pre_done.unsqueeze(0)
        gae_coeffs = (
            torch.roll(gae_coeffs, 1, dims=0) * gae_lambda * (1 - pre_done_row)
            + torch.roll(gae_coeffs, 1, dims=0) * lam_ratio * pre_done_row
        ) * (1 - done_row)
        gae_coeffs[0] = 1.0
        mask = (torch.arange(T + 1) < ii + 1).to(h_seq.dtype).unsqueeze(1)
        done_inf = done_row * float('inf')
        done_inf = torch.where(torch.isnan(done_inf), torch.zeros_like(done_inf), done_inf)
        disc_to_h = (1 - gamma) * h_seq[ii].unsqueeze(0) + gamma * (value_table + done_inf)
        Vhs_row = torch.minimum(h_seq[ii].unsqueeze(0), disc_to_h)
        Vhs_row = mask * Vhs_row
        normed = gae_coeffs / gae_coeffs.sum(dim=0, keepdim=True).clamp_min(1e-8)
        q_targets[ii] = (Vhs_row * normed).sum(dim=0)
        Vhs_row = torch.roll(Vhs_row, 1, dims=0)
        Vhs_row[0] = Vhs_seq[ii + 1]
        value_table = Vhs_row
        pre_done = done_row.squeeze(0)
    return q_targets - Vhs_seq[:-1], q_targets


def test_indexs3_basic():
    reward, energy, h_seq, _, _, lv, lvr = make_test_data()
    indexs, done = calculate_indexs3(0.99, reward, energy, h_seq, lv, lvr)
    T, nh = reward.shape
    assert indexs.shape == (T, nh)
    assert done.shape == (T + 1, nh)
    assert torch.all((done == 0) | (done == 1))
    assert torch.all(done.sum(dim=0) >= 1)
    assert torch.all(indexs >= 0) and torch.all(indexs < T + 1)


def test_indexs3_matches_reference():
    reward, energy, h_seq, _, _, lv, lvr = make_test_data(T=6, nh=3, seed=7)
    idx, done = calculate_indexs3(0.99, reward, energy, h_seq, lv, lvr)
    ref_idx, ref_done = reference_calculate_indexs3(0.99, reward, energy, h_seq, lv, lvr)
    assert torch.equal(idx, ref_idx)
    assert torch.equal(done, ref_done)


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
    done = torch.zeros(h_seq.shape[0] - 1, h_seq.shape[1])
    adv, targets = calculate_reach_gae(0.99, 0.95, h_seq, Vhs_seq, done)
    T = h_seq.shape[0] - 1
    assert adv.shape == (T, h_seq.shape[1])
    assert targets.shape == (T, h_seq.shape[1])
    assert torch.all(torch.isfinite(adv))


def test_reach_gae_matches_reference():
    _, _, h_seq, _, Vhs_seq, _, _ = make_test_data(T=7, nh=2, seed=11)
    done = torch.zeros(h_seq.shape[0] - 1, h_seq.shape[1])
    done[2, 0] = 1.0
    done[5, 1] = 1.0
    adv, targets = calculate_reach_gae(0.99, 0.95, h_seq, Vhs_seq, done)
    ref_adv, ref_targets = reference_calculate_reach_gae(0.99, 0.95, h_seq, Vhs_seq, done)
    assert torch.allclose(adv, ref_adv, atol=1e-6)
    assert torch.allclose(targets, ref_targets, atol=1e-6)


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
        test_indexs3_basic, test_indexs3_matches_reference,
        test_energy_gae_basic, test_reach_gae_basic, test_reach_gae_matches_reference,
        test_edge_all_done, test_edge_no_done, test_edge_first_done, test_edge_last_done,
        test_gamma_one, test_lam_zero
    ]
    for t in tests:
        t()
        print(f"✅ {t.__name__}")
    print(f"\n🎉 所有 {len(tests)} 个回归测试通过！")
