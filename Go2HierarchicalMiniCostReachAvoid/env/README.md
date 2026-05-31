# env/ 目录说明

本目录包含所有训练环境的实现，基于 gymnax/brax 物理引擎，为 EC-EFPPO 算法提供统一的 Reach-Avoid 环境接口。

## 文件总览

| 文件/目录 | 功能 |
|---|---|
| `env_list.py` | 环境工厂函数，根据配置名创建并包装环境 |
| `wrappers.py` | 环境包装器集合（观测变换、归一化、日志等） |
| `reach_avoid/` | 各环境的具体实现 |
| `wind_field_2.npz` | WindField 环境的风场数据（u/v 方向风速 + 障碍物信息） |
| `nv_humanoid.xml` | MuJoCo Humanoid 模型定义（备用，当前环境未使用） |
| `__init__.py` | 空文件 |

## env_list.py — 环境工厂

`get_env(config)` 函数根据 `config["EXP_NAME"]` 创建对应的环境实例，并用 `TransformObservation` 包装做观测归一化。

| EXP_NAME | 环境类 | 观测维度 | 归一化方式 |
|---|---|---|---|
| `PendulumConstraint` | `PendulumConstraint` | 4 | `[0,0,0,400]` 偏移，`[1,1,1,400]` 缩放 |
| `HopperAvoidCeiling` | `HopperAvoidCeiling` | 14 | 偏移 `[0,...,0,400]`，缩放 `[1,...,1,400]` |
| `HalfCheetahAvoid` | `HalfCheetahAvoid` | 20 | 偏移 `[2.5,0,...,0,400]`，缩放 `[3,1,...,1,400]` |
| `WindField` | `WindField` | 14 | 偏移 `[0,0,0,0,...,0,400]`，缩放 `[3,3,2,1,...,1,400]` |
| `F16Avoid` | `F16Avoid` | 26 | 偏移 `[0,...,0,400]`，缩放 `[1,...,1,400]` |

所有环境的观测末尾两维均为 `[avoid_flag, energy]`，归一化中 energy 除以 400 使其大致落在 `[-1, 1]` 范围内。

## reach_avoid/ — 环境实现

### 统一数据结构

所有环境共享相同的 `EnvState` 结构：

```
EnvState
├── state / theta      → 物理状态（各环境不同）
├── time               → 当前时间步
├── energy             → 剩余能量预算（初始随机，随消耗递减）
├── reach              → reach 值（负值=接近目标，正值=远离目标）
└── avoid              → avoid 标志（+1=安全，-1=已进入危险区域）
```

### 统一接口

每个环境实现以下方法（遵循 gymnax 接口规范）：

- `reset(key, params)` → 返回 `(obs, EnvState)`
- `step(key, state, action, params)` → 返回 `(obs, new_state, reward, done, info)`
- `action_space(params)` / `observation_space(params)` → 空间定义
- `is_reach(...)` → 计算 reach 值（各环境逻辑不同）
- `is_avoid(...)` / `is_terminal(...)` → 判定条件

**reward 语义**：所有环境的 reward 即为能量消耗（非负），训练目标是在有限初始能量下到达目标。

### 环境详情

#### `pendulum_constraint.py` — 摆锤约束

- **物理引擎**：gymnax 原生
- **状态**：`theta`（角度）、`theta_dot`（角速度）
- **观测**：`[cos θ, sin θ, θ̇, energy]`（4维）
- **动作**：1维力矩，经 `tanh` 压缩
- **reach 条件**：摆锤摆过最高点（`θ·θ_new < 0`）
- **能量消耗**：力矩超过 `torque_limit=0.1` 时，消耗 `|u|² × 8`
- **终止条件**：达到 `max_steps=2000`

#### `wind_field.py` — 风场导航

- **物理引擎**：gymnax 原生 + 自定义四旋翼动力学
- **状态**：12维（位置 xyz、姿态 ψθϕ、线速度 uvw、角速度 pqr）
- **观测**：`[12维物理状态, avoid_flag, energy]`（14维）
- **动作**：3维目标偏移，经 `tanh` 后映射为四旋翼参考位置
- **控制架构**：高层 RL → `u_ref()` 求解 LQR 推力 → RK4 积分物理状态
- **风场扰动**：从 `wind_field_2.npz` 加载风速网格，按位置索引施加外力
- **reach 条件**：到达目标区域（几何距离判定）
- **avoid 条件**：进入障碍物区域
- **终止条件**：达到 `max_steps=5000`

#### `hopper_avoid_ceiling.py` — Hopper 跳跃避障

- **物理引擎**：brax（通过 `HopperRandom` 封装）
- **状态**：brax 的 `State`（关节位置/速度 + 扩展的 energy/reach/avoid）
- **观测**：`[brax obs, avoid_flag, energy]`（14维）
- **动作**：3维关节力矩，经 `tanh` 压缩
- **reach 条件**：头部位置越过目标 x 坐标
- **avoid 条件**：头部 y 坐标超过天花板高度
- **能量消耗**：各关节力矩超限部分的平方和
- **`HopperRandom`**：继承 brax `Hopper`，reset 时随机化初始 x 位置（`+ uniform(0, 2)`）

#### `half_cheetah_avoid.py` — HalfCheetah 避障

- **物理引擎**：brax（通过 `HalfCheetahRandom` 封装）
- **状态**：brax 的 `State`
- **观测**：`[brax obs, avoid_flag, energy]`（20维）
- **动作**：6维关节力矩，经 `tanh` 压缩
- **reach 条件**：头部位置越过目标 x 坐标
- **avoid 条件**：前脚或后脚进入禁区
- **能量消耗**：`sum(u²)`
- **`HalfCheetahRandom`**：继承 brax `Halfcheetah`，reset 时随机化初始 x 位置（`+ uniform(0, 4)`）

#### `F16_avoid.py` — F16 战机避障

- **物理引擎**：gymnax 原生 + F16 非线性动力学
- **状态**：24维（位置、姿态、速度、角速度等）
- **观测**：`[24维物理状态, avoid_flag, energy]`（26维）
- **reach / avoid**：基于位置的几何判定

#### 确定性变体

- `HopperAvoidCeilingDeterministic`
- `HalfCheetahAvoidDeterministic`

与随机版本的区别：reset 时不加随机噪声，用于测试和评估。

## wrappers.py — 环境包装器

基于 `GymnaxWrapper` 基类，提供可组合的环境变换：

| 包装器 | 功能 | 是否在训练中使用 |
|---|---|---|
| `TransformObservation` | 对观测施加自定义变换（如归一化） | ✅ 所有环境均使用 |
| `FlattenObservation` | 将多维观测展平为一维 | ❌ |
| `LogWrapper` | 记录 episode 累计回报和长度 | ❌ |
| `ClipAction` | 裁剪动作到指定范围 | ❌ |
| `TransformReward` | 对 reward 施加自定义变换 | ❌ |
| `VecEnv` | 用 `jax.vmap` 向量化环境 | ❌（训练中直接用 `vmap`） |
| `NormalizeVecObservation` | 在线运行均值/方差归一化 | ❌ |
| `NormalizeVecReward` | 在线 reward 归一化 | ❌ |

主训练流程仅使用 `TransformObservation`，其余包装器为扩展预留。

## 文件间调用关系

```
EC-EFPPO.py
└── env_list.py::get_env(config)
    ├── PendulumConstraint    ← reach_avoid/pendulum_constraint.py
    ├── HopperAvoidCeiling    ← reach_avoid/hopper_avoid_ceiling.py
    │   └── HopperRandom      ← reach_avoid/hopper_random.py (继承 brax Hopper)
    ├── HalfCheetahAvoid      ← reach_avoid/half_cheetah_avoid.py
    │   └── HalfCheetahRandom ← reach_avoid/half_cheetah_random.py (继承 brax Halfcheetah)
    ├── WindField             ← reach_avoid/wind_field.py (加载 wind_field_2.npz)
    ├── F16Avoid              ← reach_avoid/F16_avoid.py
    └── TransformObservation  ← wrappers.py（观测归一化）
```
