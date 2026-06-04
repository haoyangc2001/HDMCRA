"""
Running Mean/Std 实现，用于观测归一化。

基于 Welford 在线算法，支持 GPU 加速。
参考：Stable-Baselines3 VecNormalize、CleanRL 实现。
"""

import torch
import numpy as np


class RunningMeanStd:
    """
    在线计算均值和标准差（Welford 算法）。

    用于观测归一化，支持：
    - 在线更新统计量
    - 归一化观测值
    - 保存/恢复状态

    Args:
        shape: 观测形状（如 (41,) 表示 41 维观测）
        epsilon: 数值稳定性常数
        device: 计算设备
    """

    def __init__(self, shape=(), epsilon=1e-5, device="cpu"):
        self.shape = shape
        self.epsilon = epsilon
        self.device = torch.device(device)

        # 统计量
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float64, device=self.device)

    def update(self, x: torch.Tensor) -> None:
        """
        更新统计量。

        Args:
            x: 输入张量 [..., *shape]，将在前几维上计算统计量
        """
        # 确保输入在正确设备上
        x = x.to(self.device)

        # 将输入展平为 [N, *shape]
        batch_shape = x.shape[:-len(self.shape)] if len(self.shape) > 0 else x.shape
        x_flat = x.reshape(-1, *self.shape)

        # 计算批次统计量
        batch_mean = x_flat.mean(dim=0)
        batch_var = x_flat.var(dim=0, unbiased=False)
        batch_count = x_flat.shape[0]

        # Welford 在线更新
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # 更新均值
        new_mean = self.mean + delta * batch_count / total_count

        # 更新方差
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count

        # 应用更新
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip_range: float = 10.0) -> torch.Tensor:
        """
        归一化输入。

        Args:
            x: 输入张量
            clip_range: 裁剪范围，归一化后裁剪到 [-clip_range, clip_range]

        Returns:
            归一化后的张量
        """
        x = x.to(self.device)
        normalized = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(normalized, -clip_range, clip_range)

    def state_dict(self) -> dict:
        """保存状态。"""
        return {
            "mean": self.mean.cpu(),
            "var": self.var.cpu(),
            "count": self.count.cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """恢复状态。"""
        self.mean = state_dict["mean"].to(self.device)
        self.var = state_dict["var"].to(self.device)
        self.count = state_dict["count"].to(self.device)

    def __repr__(self) -> str:
        return (
            f"RunningMeanStd(shape={self.shape}, "
            f"mean={self.mean.mean():.3f}, "
            f"std={torch.sqrt(self.var).mean():.3f}, "
            f"count={self.count.item():.0f})"
        )


class RunningMeanStdNumpy:
    """
    Numpy 版本的 Running Mean/Std，用于 CPU 端统计。

    与 PyTorch 版本功能相同，但使用 Numpy 实现，
    适合在不使用 GPU 的场景或需要与 JAX 交互时使用。
    """

    def __init__(self, shape=(), epsilon=1e-5):
        self.shape = shape
        self.epsilon = epsilon

        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """更新统计量。"""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """归一化输入。"""
        x = np.asarray(x, dtype=np.float64)
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -clip_range, clip_range)


def test_running_mean_std():
    """测试 RunningMeanStd 实现。"""
    print("Testing RunningMeanStd...")

    # 测试基本功能
    rms = RunningMeanStd(shape=(3,))

    # 生成测试数据
    torch.manual_seed(42)
    data = torch.randn(100, 3) * 10 + 5  # mean=5, std=10

    # 更新统计量
    rms.update(data)

    print(f"  真实均值: {data.mean(dim=0).tolist()}")
    print(f"  估计均值: {rms.mean.tolist()}")
    print(f"  真实标准差: {data.std(dim=0).tolist()}")
    print(f"  估计标准差: {torch.sqrt(rms.var).tolist()}")

    # 测试归一化
    normalized = rms.normalize(data)
    print(f"  归一化后均值: {normalized.mean(dim=0).tolist()}")
    print(f"  归一化后标准差: {normalized.std(dim=0).tolist()}")

    # 测试保存/恢复
    state = rms.state_dict()
    rms2 = RunningMeanStd(shape=(3,))
    rms2.load_state_dict(state)

    assert torch.allclose(rms.mean, rms2.mean), "Mean mismatch after load"
    assert torch.allclose(rms.var, rms2.var), "Var mismatch after load"
    print("  保存/恢复测试: PASSED")

    # 测试极端值
    extreme_data = torch.tensor([[1000.0, -1000.0, 0.0]])
    normalized_extreme = rms.normalize(extreme_data, clip_range=10.0)
    print(f"  极端值归一化: {normalized_extreme.tolist()}")
    assert normalized_extreme.abs().max() <= 10.0, "Clip range violated"
    print("  极端值裁剪测试: PASSED")

    print("All tests PASSED!")


if __name__ == "__main__":
    test_running_mean_std()
