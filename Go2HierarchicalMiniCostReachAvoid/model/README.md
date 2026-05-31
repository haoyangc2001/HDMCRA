# model/ 目录说明

本目录包含所有神经网络架构的定义，基于 Flax（Linen）构建，供 `rl/` 目录中的训练代码调用。

## 文件总览

| 文件 | 功能 |
|---|---|
| `actorcritic.py` | 全部网络架构定义 |
| `__init__.py` | 空文件，标记为 Python 包 |

## 网络架构一览

`actorcritic.py` 定义了 9 个 `nn.Module` 子类，按用途可分为四组：

### 第一组：EC-EFPPO 训练使用（主训练流程实际调用）

| 类名 | 功能 | 网络结构 | 输出 |
|---|---|---|---|
| `Policy_Network` | 连续动作策略网络 | 2层×256 MLP → `MultivariateNormalDiag` | 对角高斯分布 |
| `Policy_Network_Discrete` | 离散动作策略网络 | 2层×256 MLP → `Categorical` | 类别分布 |
| `Value_Network` | Value function（同时用于 Energy 和 Reach） | 2层×256 MLP → 标量 | 标量值 |

这三个类在 `EC-EFPPO.py` 中被实例化三次：Policy 网络一个，Value 网络两个（分别学习 energy value 和 reach value）。

**共同特点：**
- 隐藏层宽度均为 256
- 核权重使用 `orthogonal(sqrt(2))` 初始化，偏置初始化为 0
- 最后一层使用 `orthogonal(0.01)`（actor）或 `orthogonal(1.0)`（critic）
- 支持 `tanh`（默认）和 `relu` 两种激活函数

### 第二组：标准 Actor-Critic（对照实验）

| 类名 | 功能 |
|---|---|
| `ActorCritic_Continuous` | 连续动作的 Actor-Critic 联合网络 |
| `ActorCritic_Discrete` | 离散动作的 Actor-Critic 联合网络 |

与第一组的区别：actor 和 critic 共享同一个类，输出 `(pi, value)` 元组。在 `_ppo_update` 中使用，作为标准 PPO 的对照基线。

### 第三组：SAC 相关（扩展模块）

| 类名 | 功能 |
|---|---|
| `Actor_Network_SAC` | SAC 策略网络，带 `tanh` squashing 和自动温度调节 |
| `Representation_Network_SAC` | 表征学习网络，将 `(state, action)` 和 `future_state` 编码到同一潜空间 |

`Actor_Network_SAC` 实现了 SAC（Soft Actor-Critic）的随机策略，输出带 `tanh` 压缩的动作，支持可学习的 `log_std` 范围裁剪。未在当前训练流程中使用。

### 第四组：IQE 距离度量（扩展模块）

| 类名 | 功能 |
|---|---|
| `IQE` | Implicit Quantile Embedding，计算两个分布的嵌入距离 |
| `MaxMean` | 距离聚合层，用可学习的 α 混合 max 和 mean |

`IQE` 实现了论文中的区间覆盖距离（Interval Quantile Embedding）算法：将输入分量排序后合并区间，计算不相交区间的总长度。`MaxMean` 将多分量距离聚合为标量。这两个类用于表示学习的度量模块，未在主训练流程中使用。

## 调用关系

```
EC-EFPPO.py
├── Policy_Network          → 训练时采样动作
├── Policy_Network_Discrete → 离散环境的动作采样
└── Value_Network           → 实例化两次
    ├── train_state_energy  → 估计能量消耗 V(s)
    └── train_state_h       → 估计 reach 值 h(s)

EFPPO_utils.py
├── Policy_Network.apply()  → _env_step 中采样
└── Value_Network.apply()   → _env_step 中估计 value

root_finding.py
├── train_state_energy.apply_fn() → 二分法中评估能量 value
└── train_state_h.apply_fn()      → 二分法中评估 reach value
```
