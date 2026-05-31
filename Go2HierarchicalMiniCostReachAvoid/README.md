# EC-EFPPO: Energy-Constrained Earliest Feasible PPO

基于 JAX/Flax 实现的最小代价 Reach-Avoid 强化学习算法。在保证安全到达目标（reach）和避免障碍物（avoid）的前提下，最小化能量消耗。

## 目录结构

```
Go2HierarchicalMiniCostReachAvoid/
├── rl/                    # 强化学习算法核心
│   ├── EC-EFPPO.py        # 训练主入口
│   ├── EFPPO_utils.py     # 环境交互与 PPO 更新
│   ├── gae.py             # GAE 优势函数（含 Reach-Avoid 变体）
│   ├── arguments.py       # 命令行参数定义
│   ├── root_finding.py    # 二分法根查找（推理阶段）
│   └── utils.py           # 优化器、可视化等工具函数
├── model/                 # 神经网络架构
│   └── actorcritic.py     # Policy / Value / SAC / IQE 网络定义
├── env/                   # 训练环境
│   ├── env_list.py        # 环境工厂函数
│   ├── wrappers.py        # 环境包装器（观测归一化等）
│   └── reach_avoid/       # 各环境的具体实现
│       ├── pendulum_constraint.py
│       ├── hopper_avoid_ceiling.py
│       ├── half_cheetah_avoid.py
│       ├── wind_field.py
│       └── F16_avoid.py
└── script/                # 训练启动脚本
    ├── run_pendulum_constraint.sh
    ├── run_hopper_reach.sh
    ├── run_hopper_avoid_ceiling.sh
    ├── run_half_cheetah_avoid.sh
    ├── run_wind_field.sh
    └── run_f16_avoid.sh
```

各子目录的详细说明见对应目录下的 README。

## 算法概述

### 要解决的问题

给定一个带有障碍物的环境和有限的初始能量预算，训练智能体：

1. **Reach**：到达目标区域
2. **Avoid**：全程避开障碍物
3. **Min-Cost**：在满足前两条的前提下，能量消耗尽可能小

### 核心思想

#### 双 Value Function 架构

训练两个独立的 value network：

- **Energy Value V(s)**：估计从当前状态出发的累计能量消耗
- **Reach Value h(s)**：估计当前状态离"安全到达目标"的距离（来自 Hamilton-Jacobi reachability 理论）

两者组合为：

```
V_total(s) = max( h(s), V(s) - energy_budget )
```

`max` 操作体现了优先级：先确保安全到达，再优化能量。

#### Earliest Reach Index

算法反向扫描轨迹，找到使组合目标 `max(V_s - e, V_h)` 最小的时间步，作为"最佳到达时机"。这取代了传统 PPO 中依赖环境自然终止的 done 机制，由 `calculate_indexs3` 实现。

#### Reach-Avoid GAE

`calculate_gae_reach4` 实现了专用的广义优势估计：

- reach value 的 DP 递推使用 `min(h, γ·V_h)`（Bellman 收缩性质）
- done 边界处的 GAE 系数用 `λ/(1-λ)` 重新归一化，避免梯度信号断裂

#### 三网络独立更新

每次 PPO 更新分别计算三组梯度：

- **Policy** ← 组合优势（reach + energy）
- **Energy Value** ← energy GAE targets
- **Reach Value** ← reach GAE targets

## 实验环境

| 环境 | 动作维度 | 观测维度 | 物理引擎 | 任务描述 |
|---|---|---|---|---|
| PendulumConstraint | 1 | 4 | gymnax | 摆锤在受限力矩下摆起 |
| HopperAvoidCeiling | 3 | 14 | brax | 跳跃时避免头部碰天花板 |
| HalfCheetahAvoid | 6 | 20 | brax | 奔跑时避免肢体进入禁区 |
| WindField | 3 | 14 | gymnax + 自定义动力学 | 四旋翼在风场中导航避障 |
| F16Avoid | — | 26 | gymnax | F16 战机避障飞行 |

所有环境的 reward 即为能量消耗（非负），观测末尾拼接 `[avoid_flag, energy]`。

## 快速开始

### 依赖

```
jax
flax
optax
gymnax
brax
distrax
einops
wandb
matplotlib
colour
control  (python-control)
```

### 训练

从项目根目录运行：

```bash
# 摆锤约束环境
bash script/run_pendulum_constraint.sh

# Hopper 避障
bash script/run_hopper_avoid_ceiling.sh

# HalfCheetah 避障
bash script/run_half_cheetah_avoid.sh

# 风场导航
bash script/run_wind_field.sh

# F16 避障
bash script/run_f16_avoid.sh
```

训练产物（checkpoint、可视化图片）保存到 `model/{DIR}/`，日志同步上报到 wandb（项目名 `EC-EFPPO-{EXP_NAME}`）。

### 关键超参数

| 参数 | 含义 | 典型值 |
|---|---|---|
| `GAMMA_ENERGY` | 能量 value function 折扣因子 | 1.0（无折扣） |
| `GAMMA_REACH_INIT/FINAL` | reach 函数折扣因子退火范围 | 0.99 / 0.99 |
| `NUM_ENVS` | 并行环境数 | 32 ~ 2048 |
| `TOTAL_TIMESTEPS` | 总训练步数 | 20M ~ 80M |
| `STEP_SCAN` | 内层训练循环次数 | 1 ~ 40 |

完整参数说明见 `rl/README.md` 和 `script/README.md`。

## 与 Go2HierarchicalReachAvoidRL 的关系

本项目是 EC-EFPPO 算法的 **JAX 原始参考实现**。`Go2HierarchicalReachAvoidRL` 将同一算法移植到 PyTorch，适配 Unitree Go2 真实机器人，核心数学逻辑（earliest reach index、reach-avoid GAE、三网络更新）均源自本项目。
