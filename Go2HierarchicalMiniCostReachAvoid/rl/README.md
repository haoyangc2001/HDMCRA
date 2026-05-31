# rl/ 目录说明

本目录包含 EC-EFPPO（Energy-Constrained Earliest Feasible PPO）算法的核心实现，基于 JAX/Flax，是整个项目的强化学习训练引擎。

## 文件总览

| 文件 | 功能 |
|---|---|
| `EC-EFPPO.py` | 训练主入口，包含网络初始化、训练循环、日志和可视化 |
| `EFPPO_utils.py` | 环境交互步和 PPO 更新的核心函数 |
| `gae.py` | GAE（广义优势估计）及 Reach-Avoid 专用优势函数的多种变体 |
| `arguments.py` | 命令行参数定义 |
| `root_finding.py` | 二分法根查找，用于推理阶段确定最小能量预算阈值 |
| `utils.py` | 优化器配置、颜色映射、树索引等工具函数 |
| `__init__.py` | 空文件，标记为 Python 包 |

## 文件详细说明

### `EC-EFPPO.py` — 训练主入口

训练的顶层脚本，负责组装所有组件并驱动训练循环。

**核心类：**

- `TrainState`：继承 `flax.training.train_state.TrainState`，额外维护 `mean`、`variance`、`count` 字段用于在线归一化。

**核心函数：**

- `train(env, env_params, config, rng)`：主训练函数，内部嵌套定义了 `_train` 作为单次更新的 JIT 编译单元。

**训练流程：**

1. 初始化三个网络（Policy、Energy Value、Reach Value）及其 `TrainState`
2. 外层循环按 `STEP_SCAN` 分组迭代，内层调用 JIT 编译的 `_train`
3. 每次 `_train` 内部：
   - 用 `jax.lax.scan` + `env_step` 收集 `NUM_STEPS` 步轨迹
   - 计算三路 GAE 优势（reach、energy、combined）
   - 用 `calculate_indexs3` 计算最早可达时间步的 done 标志
   - 用 `jax.lax.scan` + `update_epoch` 执行 `UPDATE_EPOCHS` 轮 PPO 更新
4. 外层循环中记录 wandb 日志、保存 checkpoint、生成可视化

**γ 衰减调度：** `GAMMA_REACH` 从 `INIT` 线性退火到 `FINAL`，两路分别以 1x 和 2x 速率退火。

---

### `EFPPO_utils.py` — 环境交互与 PPO 更新

**`_env_step(env, env_params, runner_state, _)`**

单步环境交互函数，设计为 `jax.lax.scan` 的 body：
1. 用 Policy 网络采样动作
2. 用 Energy/Reach Value 网络估计当前 value
3. `jax.vmap(env.step)` 并行执行多环境
4. 将转移存入 `Transition_reach` 命名元组

**`_ecefppo_update(config, update_state, ent)`**

EC-EFPPO 的核心更新函数，与标准 PPO 的关键区别：

- 维护三个独立的 `TrainState`（policy、energy、reach），分别计算梯度并更新
- **Policy loss**：标准 PPO clipped objective + 熵正则化，优势信号来自 `advantages_total`（reach 和 energy 的组合优势）
- **Energy value loss**：clipped value loss，目标来自 `targets_V`
- **Reach value loss**：clipped value loss，目标来自 `targets_h`
- 数据先 shuffle 再分 minibatch，用 `jax.lax.scan` 遍历

**`_ppo_update(config, update_state, ent)`**

标准 PPO 更新（对照实验用），单一网络同时输出 policy 和 value，未在主训练流程中使用。

---

### `gae.py` — 优势函数计算

本文件是算法的数学核心，实现了多种 GAE 变体以适配 Reach-Avoid 框架。

**数据结构：**

- `Transition_reach`：命名元组，存储 `(done, action, value, value_reach, reward, energy, log_prob, obs, info, reach)`

**标准 GAE 函数：**

- `calculate_advantage(gae_nval_gamma_lambda, transition)` → 单步 GAE 递推
- `calculate_gae(gamma, gae_lambda, value, reward, done, last_value)` → 标准 GAE（反向 scan）
- `calculate_advantage2(...)` / `calculate_gae2(...)` → 带 done mask 的 GAE 变体，用于 energy value function

**Reach-Avoid GAE 函数：**

- `calculate_gae_reach(gamma, gae_lambda, Tp1_hs, Tp1_Vhs)` → 基础版 reach GAE，用 `min(h, γ·V_h)` 做 DP
- `calculate_gae_reach2(...)` → 加入 `(1-γ)·h + γ·V_h` 折扣形式
- `calculate_gae_reach3(...)` → 加入 done 标志处理，done 时用 `+inf` 惩罚
- `calculate_gae_reach4(...)` → **最终使用版本**，GAE 系数在 done 边界处用 `λ/(1-λ)` 重新归一化，处理 done 前后系数断裂问题

**Earliest Reach Index 计算：**

- `calculate_indexs(gamma, reward, energy, T_hs)` → 基础版
- `calculate_indexs2(...)` → 加入 `max(V_s - energy, V_h)` 信号
- `calculate_indexs3(...)` → **最终使用版本**，额外考虑 bootstrap value（`last_value`、`last_value_reach`），反向扫描找到使组合目标最小的时间步索引

**其他：**

- `calculate_gae3(...)` → 基于 index 的分段 GAE，用于对照实验

---

### `arguments.py` — 命令行参数

`get_args(args)` 函数定义所有训练参数，主要分四类：

- **实验配置**：`EXP_NAME`、`DIR`、`NAME`、`SEED`、`CUDA_USE`
- **PPO 超参**：`LR`、`CLIP_EPS`、`GAE_LAMBDA`、`ENT_COEF`、`VF_COEF`、`MAX_GRAD_NORM`、`UPDATE_EPOCHS`、`NUM_MINIBATCHES`
- **能量约束参数**：`GAMMA_ENERGY`、`GAMMA_REACH_INIT`、`GAMMA_REACH_FINAL`、`LAMBDA_REACH`、`THRESHOLD_CPPO`、`K_P`
- **训练规模**：`NUM_ENVS`、`NUM_STEPS`、`TOTAL_TIMESTEPS`、`STEP_SCAN`
- **开关**：`ANNEAL_LR`、`ANNEAL_ENT`、`TEST_MODE`、`DISCRETE`、`FIX_LAMBDA`

---

### `root_finding.py` — 二分法根查找

**`Bisection` 类**

在训练好的 value function 上用二分法查找能量预算阈值——即给定当前状态，找到使 `max(V_E(s, e) - e, V_h(s) + threshold) = 0` 的能量值 `e`。

**核心思路：**

- 将观测的最后一维（energy）设为变量，在 `[-1, 1]` 范围内二分搜索
- 每步计算 `max(energy_net(obs) - e, reach_net(obs) + threshold)`，根据符号缩小区间
- 通过 `jax.lax.scan` 实现 JIT 兼容的迭代

**方法：**

- `init_state(lb, ub)` → 初始化上下界和单调性标志
- `run()` → 执行二分搜索，返回根的近似值
- `run_detailed()` → 同上，但返回每步的中间结果（用于可视化/调试）

---

### `utils.py` — 工具函数

- `optimizer(config)` → 构建 optax 优化器（`clip_by_global_norm` + `adam`），支持可选的学习率线性退火
- `linear_schedule(config, count)` → 学习率退火函数，线性衰减到 0
- `get_BuRd()` → 生成蓝-红双色发散 colormap（用于 value function 等高线图可视化）
- `tree_index1(tree, idx)` → 取 pytree 第一维的第 `idx` 个元素
- `tree_index2(tree, idx)` → 取 pytree 第二维的第 `idx` 个元素

## 文件间调用关系

```
EC-EFPPO.py (训练主入口)
├── arguments.py          ← get_args() 解析参数
├── EFPPO_utils.py
│   ├── _env_step()       ← 单步环境交互
│   └── _ecefppo_update() ← PPO 三网络更新
├── gae.py                ← 各种 GAE 变体
│   ├── calculate_gae_reach4()   ← reach 优势
│   ├── calculate_gae2()         ← energy 优势
│   ├── calculate_gae_reach4()   ← 组合优势（复用）
│   └── calculate_indexs3()      ← 最早可达 index
└── utils.py
    ├── optimizer()       ← 构建优化器
    ├── tree_index1/2()   ← 数据索引
    └── get_BuRd()        ← 可视化颜色

root_finding.py           ← 推理阶段独立使用，不参与训练
```
