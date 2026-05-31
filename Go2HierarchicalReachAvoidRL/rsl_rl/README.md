# rsl_rl/ 目录说明

本目录是 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)（Robot Systems Lab RL，ETH Zurich）的改造版本，提供 PyTorch 实现的 on-policy 强化学习算法框架。原始版本仅包含标准 PPO，本项目新增了 `ReachAvoidPPO`，实现了 Reach-Avoid PPO 算法，是从 JAX 参考实现（`Go2HierarchicalMiniCostReachAvoid/rl/gae.py`）移植而来。

## 目录结构

```
rsl_rl/
├── setup.py                         # pip 安装脚本（包名 rsl_rl，版本 1.0.2）
└── rsl_rl/
    ├── algorithms/                  # RL 算法
    │   ├── ppo.py                   # 标准 PPO（原始 rsl_rl）
    │   └── reach_avoid_ppo.py       # Reach-Avoid PPO（本项目新增）
    ├── modules/                     # 网络架构
    │   ├── actor_critic.py          # 前馈 Actor-Critic
    │   └── actor_critic_recurrent.py # 循环 Actor-Critic（LSTM/GRU）
    ├── runners/                     # 训练运行器
    │   └── on_policy_runner.py      # On-policy 训练循环（标准 PPO 用）
    ├── storage/                     # 经验存储
    │   └── rollout_storage.py       # Rollout 缓冲区（标准 PPO 用）
    └── utils/                       # 工具函数
        └── utils.py                 # 轨迹分割/填充工具
```

## 文件详细说明

### `algorithms/reach_avoid_ppo.py` — Reach-Avoid PPO（587行，核心新增）

本项目的核心算法实现，从 JAX 版本的 `calculate_gae_reach4` 移植到 PyTorch。包含四个组件：

#### `_calculate_reach_gae(gamma, lam, g_seq, value_seq, done_seq, h_seq)`

Reach-Avoid 专用的广义优势估计函数，算法逻辑：
- 从后向前动态规划，维护 `value_table` 存储多步回溯的价值候选
- 每步计算 `vhs_row = max(h, min(g, γ·V_table))`，实现安全优先的 Bellman 递推
- GAE 系数 `gae_coeffs` 在 done 边界处用 `λ/(1-λ)` 重新归一化，避免梯度断裂
- 最终用归一化系数加权平均得到 `Q_target`，优势 = `Q_target - V(s)`

#### `ReachAvoidBuffer`

扩展的 rollout 缓冲区，在标准 PPO 的 `observations/actions/log_probs/values/dones` 基础上，额外存储：
- `g_values`：到达目标函数值序列
- `h_values`：安全约束函数值序列

`compute_advantages()` 内部调用 `_calculate_reach_gae` 计算优势和 Q 目标。

#### `ReachAvoidBatch`

训练数据批次封装，将缓冲区数据展平为 `(horizon×num_envs, ...)` 形状，支持 mini-batch 随机采样。

#### `ReachAvoidPPO`

训练器类，与标准 PPO 的关键区别：
- 维护独立的 `ReachAvoidBuffer`（而非标准 `RolloutStorage`）
- `act()` 返回 `(actions, log_probs, values)`
- `update()` 多 epoch、多 mini-batch 迭代，使用 Reach-Avoid GAE 计算的优势
- 损失函数：`policy_loss + value_coef×value_loss - entropy_coef×entropy`
- 支持 KL 散度自适应学习率调度

### `algorithms/ppo.py` — 标准 PPO（187行，原始 rsl_rl）

原始 rsl_rl 的 PPO 实现，本项目中用于：
- 低层运动策略的预训练（通过 `OnPolicyRunner` 调用）
- 作为高层 Reach-Avoid PPO 的对照参考

核心流程与 `ReachAvoidPPO` 类似，但使用标准 `RolloutStorage` 和标准 GAE。

### `modules/actor_critic.py` — 前馈 Actor-Critic（155行）

`ActorCritic(nn.Module)`：
- **Actor**：MLP → 高斯分布 `Normal(mean, std)`，`std` 为可学习参数
- **Critic**：独立 MLP → 标量 value
- 支持 actor 和 critic 使用不同的输入维度（`num_actor_obs` vs `num_critic_obs`，用于 asymmetric training）
- 默认隐藏层 `[256, 256, 256]`，激活函数 `elu`
- `act()` → 采样动作 + log_prob
- `evaluate()` → 评估 value
- `act_inference()` → 确定性推理（用 mean 而非采样）

### `modules/actor_critic_recurrent.py` — 循环 Actor-Critic（约 60行）

`ActorCriticRecurrent(ActorCritic)`：
- 在前馈 `ActorCritic` 前增加 `Memory` 模块（LSTM 或 GRU）
- Actor 和 Critic 各自独立的 RNN 记忆
- 支持序列分割和填充（处理 episode 边界）

`Memory(nn.Module)`：封装 LSTM/GRU，管理隐藏状态的重置和传递。

### `runners/on_policy_runner.py` — On-policy 训练循环（233行）

`OnPolicyRunner`：标准 PPO 的训练运行器，本项目中**仅用于低层运动策略的预训练**。

核心流程：
1. 创建 `ActorCritic` 网络和 `PPO` 算法实例
2. 初始化 `RolloutStorage`
3. 循环：收集 rollout → 计算 advantages → `alg.update()` → 日志/保存

支持：
- TensorBoard 日志
- checkpoint 保存和恢复
- 自适应学习率（KL 散度调度）

高层 Reach-Avoid 训练不使用此类，而是由 `legged_gym_go2/legged_gym/scripts/train_reach_avoid.py` 中的自定义训练循环驱动。

### `storage/rollout_storage.py` — Rollout 缓冲区（234行）

`RolloutStorage`：标准 PPO 的经验回放缓冲区，存储：
- `observations` / `critic_observations`（支持 asymmetric training）
- `actions` / `actions_log_prob` / `action_mean` / `action_sigma`
- `rewards` / `dones` / `values`
- `hidden_states`（RNN 用）

`compute_returns()` 使用标准 GAE 计算 advantages 和 returns。

### `utils/utils.py` — 工具函数

- `split_and_pad_trajectories(tensor, dones)` — 在 done 处分割轨迹并填充，用于 RNN 处理 episode 边界
- `unpad_trajectories(trajectories, masks)` — 移除填充，恢复原始数据

## 与 JAX 参考实现的对应关系

| JAX (`Go2HierarchicalMiniCostReachAvoid/rl/`) | PyTorch (`rsl_rl/`) |
|---|---|
| `gae.py::calculate_gae_reach4` | `reach_avoid_ppo.py::_calculate_reach_gae` |
| `gae.py::Transition_reach` | `reach_avoid_ppo.py::ReachAvoidBuffer` |
| `EFPPO_utils.py::_ecefppo_update` | `reach_avoid_ppo.py::ReachAvoidPPO.update` |
| `EC-EFPPO.py::train` | 由 `train_reach_avoid.py` 自定义循环替代 |
| `model/actorcritic.py::Policy_Network` | `actor_critic.py::ActorCritic.actor` |
| `model/actorcritic.py::Value_Network` | `actor_critic.py::ActorCritic.critic` |

## 与上游 rsl_rl 的改动

原始 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 仅包含标准 PPO。本项目新增：

1. **`algorithms/reach_avoid_ppo.py`** — 全新文件，实现 Reach-Avoid PPO 及配套的 Buffer/Batch
2. 其余文件（`ppo.py`、`actor_critic.py`、`on_policy_runner.py`、`rollout_storage.py`）保持上游不变，用于低层策略预训练

## 调用关系

```
train_reach_avoid.py (高层导航训练)
└── ReachAvoidPPO  ←  reach_avoid_ppo.py
    ├── _calculate_reach_gae()  ← JAX gae.py::calculate_gae_reach4 移植
    ├── ReachAvoidBuffer        ← 存储 g_values + h_values
    └── ActorCritic             ← actor_critic.py（仅用于高层策略）

OnPolicyRunner (低层运动预训练)
└── PPO  ←  ppo.py
    ├── RolloutStorage          ← rollout_storage.py
    └── ActorCritic             ← actor_critic.py（或 ActorCriticRecurrent）

deploy/deploy_real/deploy_real.py (真实部署)
└── torch.jit.load()            ← 加载 OnPolicyRunner 保存的 checkpoint
```
