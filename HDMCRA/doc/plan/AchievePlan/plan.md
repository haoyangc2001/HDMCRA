# HDMCRA 项目实施方案

## 整体架构

在 `/home/caohy/repositories/HDMCRA/HDMCRA` 下搭建完全独立的项目，将 `Go2HierarchicalReachAvoidRL` 中的所有必要组件（IsaacGym、rsl_rl、legged_gym_go2）完整复制进来，确保项目不依赖任何外部仓库。在此基础上，将高层导航算法从 Reach-Avoid PPO 替换为 `/home/caohy/repositories/HDMCRA/Go2HierarchicalMiniCostReachAvoid`  EC-EFPPO（Energy-Constrained Earliest Feasible PPO）。核心改动集中在环境层（加入 energy 状态）和算法层（三网络 + earliest reach index），底层 IsaacGym 仿真和低层运动策略完全复用。

---

## 第一步：搭建项目骨架与开发环境

### 目标

建立可独立运行的项目结构，创建 conda 虚拟环境，安装所有依赖，并确保能跑通原有的 Reach-Avoid PPO 训练。

### 当前状态（2026-05-31 更新）

**已完成（从另一台设备迁移）：**
- 目录结构已复制：`Go2HierarchicalReachAvoidRL/legged_gym_go2/`、`rsl_rl/`、`isaacgym/` 已完整复制到 `HDMCRA/`
- `setup.py` 包名已修改：`unitree_rl_gym` → `hdmcr-unitree-rl-gym` (v2.0.0)，`rsl_rl` → `hdmcr-rsl-rl` (v2.0.0)
- `legged_gym_go2/setup.py` 的 `install_requires` 已更新为依赖 `hdmcr-rsl-rl`
- `numpy` 版本约束从 `==1.20` 放宽为 `>=1.20`

**待完成（本台设备环境搭建）：**
- 创建 conda 环境 `hdmcr` (Python 3.8)
- 安装 PyTorch 1.13.1 + CUDA 11.6（或 PyTorch 2.0.1 + CUDA 11.8）
- 安装 isaacgym（关键风险：gymtorch 编译）
- 安装 rsl_rl、legged_gym_go2
- 验证 import 和 GPU 仿真

### 环境评估（本台设备）

| 项目 | 本台设备 | 原计划要求 | 兼容性 |
|---|---|---|---|
| GPU | NVIDIA RTX 4090 (24GB) | NVIDIA GPU | ✅ 兼容 |
| CUDA | CUDA 12.6 | CUDA 11.1 | ⚠️ 不兼容 |
| Python | 3.13 (base) / 3.11 (env_isaaclab) | Python 3.8 | ❌ 不兼容 |
| PyTorch | 2.7.0+cu128 (env_isaaclab) | PyTorch 1.8.1 | ❌ 不兼容 |

**关键问题：**
1. **CUDA 版本不兼容**：本台设备 CUDA 12.6 与原计划的 CUDA 11.1 不兼容，PyTorch 1.8.1 预编译包是为 CUDA 11.1 构建的
2. **Python 版本不兼容**：IsaacGym 要求 Python <3.9，但本台设备只有 Python 3.13 和 3.11
3. **gymtorch 编译风险**：gymtorch C++ 扩展编译可能因 PyTorch 版本不兼容而失败

**解决方案：**
- 使用 PyTorch 1.13.1 + CUDA 11.6 替代原计划的 PyTorch 1.8.1 + CUDA 11.1
- 如果 PyTorch 1.13 安装失败，备选方案为 PyTorch 2.0.1 + CUDA 11.8
- 创建新的 conda 环境，使用 Python 3.8

### 具体任务

**目录结构（已完成）：**
- ✅ 将 `Go2HierarchicalReachAvoidRL/legged_gym_go2/` 完整复制到 `HDMCRA/legged_gym_go2/`
- ✅ 将 `Go2HierarchicalReachAvoidRL/rsl_rl/` 完整复制到 `HDMCRA/rsl_rl/`
- ✅ 将 `Go2HierarchicalReachAvoidRL/isaacgym/` 完整复制到 `HDMCRA/isaacgym/`
- ✅ 修改 `legged_gym_go2/setup.py` 和 `rsl_rl/setup.py` 中的包名或版本号

**环境与依赖（待完成）：**
- 创建 conda 虚拟环境，命名为 `hdmcr`，Python 版本使用 3.8
- 安装 PyTorch 1.13.1 + CUDA 11.6：
  ```bash
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
  ```
- 如果上述安装失败，尝试 PyTorch 2.0.1 + CUDA 11.8：
  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
  ```
- 安装 IsaacGym：进入 `isaacgym/python/` 目录，执行 `pip install -e .`
  - **关键风险点**：gymtorch C++ 扩展编译（`gymtorch.cpp`）可能因 PyTorch 版本不兼容而失败
  - 如果编译失败，需要手动修改 `gymtorch.py` 中的 `TORCH_MAJOR`/`TORCH_MINOR` 编译标志
- 安装 rsl_rl：进入 `rsl_rl/` 目录，执行 `pip install -e .`
- 安装 legged_gym_go2：进入 `legged_gym_go2/` 目录，执行 `pip install -e .`
- 安装其他可选依赖：`scipy`、`opencv-python`、`wandb`

**验证：**
- 依次执行 `import isaacgym`、`import rsl_rl`、`import legged_gym`，确认无导入错误
- 运行 IsaacGym 示例确认 GPU 仿真能正常启动
- 跑通一次原有的 `train_reach_avoid.py` 训练（短时间），确认环境和算法都正常工作

**故障排查：**
- 如果 gymtorch 编译失败：
  1. 检查 PyTorch CUDA 版本是否与系统 CUDA 兼容
  2. 尝试升级/降级 PyTorch 版本
  3. 手动设置 `TORCH_MAJOR`/`TORCH_MINOR` 编译标志
  4. 如果所有方案失败，考虑使用 PyTorch 2.0 + CUDA 11.8

### 工作重心

PyTorch 版本与 IsaacGym 的兼容性。本台设备 CUDA 12.6 与原计划的 CUDA 11.1 不兼容，需要使用 PyTorch 1.13 + CUDA 11.6 或 PyTorch 2.0 + CUDA 11.8。gymtorch C++ 扩展编译是关键风险点，如果编译失败需要调整 PyTorch 版本或手动修改编译标志。这一步是后续所有改造的基础。

---

## 第二步：环境层改造 — 加入 Energy 状态

### 目标

在高层导航环境中实现能量预算的初始化、消耗计算和观测拼接，使 EC-EFPPO 所需的 energy 信息能够被训练层获取。

### 修改文件

- `legged_gym/envs/go2/high_level_navigation_env.py`
- `legged_gym/envs/go2/go2_config.py`

### 具体任务

- 在 `HighLevelNavigationConfig` 类中新增能量相关参数：`min_energy`、`max_energy`（初始随机范围）、`energy_consumption_scale`（消耗缩放系数）。这些参数对应 JAX 版 `EnvParams` 中的 `min_energy=-400`、`max_energy=800`
- 在 `HighLevelNavigationEnv.__init__` 中新增两个缓冲区：`self.energy`（当前能量预算，形状 `[num_envs]`）和 `self.energy_consumption`（当前步消耗量）
- 修改 `reset()` 方法：在重置环境时，用均匀分布随机初始化每个环境的 energy 值，范围为 `[min_energy, max_energy]`
- 修改 `step()` 方法：采用简单方案计算能量消耗，即 `consumption = ||high_level_actions||² × scale`，对应 JAX 版 Pendulum 环境的 `|u|² × 8`。然后更新 `self.energy = clip(self.energy - consumption, min_energy, max_energy)`。（精确方案——从底层力矩计算消耗 `sum(torques²)`——作为后续可选改进，在第八步中实验验证）
- 修改观测维度：将 `num_high_level_obs` 增加 1，用于承载 energy 信息
- 修改 `_compute_high_level_observations()` 方法：在现有观测的末尾拼接 `self.energy.unsqueeze(1)`
- 修改 `step()` 的返回值，增加 `energy` 和 `energy_consumption`
- 在 `go2_config.py` 中的 `GO2HighLevelCfg` 中同步更新 `num_observations`（加 1）

### 工作重心

energy 状态的正确性和一致性。需要确保 energy 在 reset 时随机初始化、在 step 中正确递减、在观测中正确拼接，且不会因数值范围问题影响训练稳定性。建议将 energy 归一化到 `[-1, 1]` 范围（除以 400），与 JAX 版的做法一致。

---

## 第三步：分层环境改造 — 透传 Energy 数据

### 目标

在 `HierarchicalGO2Env` 中透传 energy 数据，使上层训练脚本能获取到完整的 EC-EFPPO 所需信息。

### 修改文件

- `legged_gym/envs/go2/hierarchical_go2_env.py`

### 具体任务

- 修改 `HierarchicalGO2Env.reset()` 的返回值，在原有的 `(high_level_obs, g_values, h_values)` 基础上增加 `energy`
- 修改 `HierarchicalGO2Env.step()` 的返回值，在原有的 `(high_level_obs, g_values, h_values, dones, infos)` 基础上增加 `energy` 和 `energy_consumption`
- 注意：本步骤只修改 `HierarchicalGO2Env`。`HierarchicalVecEnv`（定义在 `train_reach_avoid.py` 中）的适配放在第七步统一处理

### 工作重心

数据流的完整性。确保 energy 信息从底层环境一路透传到 `HierarchicalGO2Env` 的返回值中，不丢失、不错位。同时注意 `high_level_action_repeat` 场景下能量消耗的累加逻辑——每次低层执行都应贡献能量消耗。

---

## 第四步：移植 Earliest Reach Index 和三路 GAE

### 目标

将 JAX 版的核心数学算法移植为 PyTorch 实现，这是整个项目的算法核心。

### 新建文件

`rsl_rl/rsl_rl/algorithms/ecfppo_gae.py`

### 具体任务

**明确移植范围：** EC-EFPPO 只用到 `gae.py` 中的 3 个函数，不需要移植全部 10+ 个变体：
1. `calculate_indexs3`（第 334-378 行）
2. `calculate_gae2` + `calculate_advantage2`（第 379-432 行）
3. `calculate_gae_reach4`（第 195-247 行）

其中 `calculate_gae_reach4` 已有 PyTorch 版本 `_calculate_reach_gae`，需要逐行对比验证一致性。

**移植 `calculate_indexs3`：** 源代码位于 `gae.py` 第 334-378 行。该函数反向扫描轨迹，维护 `Vs_row`（能量累计值）和 `Vhs_row`（reach 值），在每个时间步计算 `V_total = max(Vs - energy, Vhs)`，然后找到使组合价值最小的时间步索引作为 done 标志。JAX 版用 `jax.lax.scan` 实现反向迭代，PyTorch 版需要用 `for` 循环或 `torch.flip` 配合向量化操作实现。关键注意点：
- done 矩阵的索引赋值 `done.at[index, jnp.arange(nh)].set(1.0)` 在 PyTorch 中对应 `done[index, torch.arange(nh)] = 1.0`
- 初始 carry 中的 `jnp.ones(...) * jnp.inf` 值在 PyTorch 中需使用 `torch.inf`，且需验证 `argmin` 对 `inf` 的行为在两个框架中一致
- `next_mask_1` 的 `jnp.roll` 操作在 PyTorch 中对应 `torch.roll`

**移植 `calculate_gae2` + `calculate_advantage2`：** 源代码位于 `gae.py` 第 379-432 行。这是带 done mask 的 GAE，用于计算 energy value function 的优势。**注意：** `calculate_advantage2` 中使用 `done` 和 `next_done` 双重 mask 控制回报传播（`delta = (reward + Gamma * next_value * (1 - next_done)) * (1 - done) - value`），这与 `_calculate_reach_gae` 的单 done mask 不同，需要独立实现一个新的 PyTorch 函数（如 `calculate_energy_gae`），**不能复用** `_calculate_reach_gae`。

**验证 `calculate_gae_reach4` 移植正确性：** 现有的 `_calculate_reach_gae` 已经从 JAX 版移植，需要逐行对比确认：
- GAE 系数在 done 边界处的 `lam/(1-lam)` 重新归一化逻辑完全一致
- `pre_done_row` 的处理——JAX 版在系数更新中同时使用了前一步 done 和当前步 done
- `Vhs_row` 的 DP 更新逻辑（`minimum(hs, disc_to_h)`）完全对齐

**环境 done 合并：** JAX 版 `calculate_indexs3` 产生的 `done` 只反映 earliest reach index，不包含环境真正终止（如摔倒 `base_contact`、超时）的信号。在 Go2 环境中 `terminate_after_contacts_on=["base"]` 会导致环境真正 reset。因此在调用 `calculate_indexs3` 之后，需要将环境 `dones` OR 到 `done` 矩阵中：`done = done | env_dones`。

**编写验证脚本：** 用小规模随机数据（如 `horizon=10, num_envs=2`）分别运行 JAX 版和 PyTorch 版，断言输出数值一致（容差 1e-5）。特别覆盖：(a) 全部 `inf` 初始值的 `argmin` 行为；(b) 第一步就 done 的情况；(c) 环境中途终止的情况。

### 工作重心

算法移植的数值正确性。`calculate_indexs3` 是最复杂的部分，涉及多维数组的反向迭代和条件索引赋值，需要特别注意 JAX 和 PyTorch 在数组操作语义上的差异。`calculate_gae2` 需要独立实现（不能复用 `_calculate_reach_gae`），因为其 `done`/`next_done` 双重 mask 逻辑不同。

---

## 第五步：实现三网络架构

### 目标

扩展现有的 `ActorCritic` 模块，实现 Policy + Energy Value + Reach Value 三网络架构。

### 修改文件

`rsl_rl/rsl_rl/modules/actor_critic.py`

### 具体任务

- 新增 `EC_EFPPO_ActorCritic(nn.Module)` 类，包含三个子网络：`self.actor`（策略网络）、`self.energy_critic`（能量价值网络）、`self.reach_critic`（reach 价值网络）

- **网络结构决策：** 每个子网络的结构为 **2层×256 MLP**，激活函数使用 **tanh**，与 JAX 版 `Policy_Network` 和 `Value_Network` 的默认结构一致。选择 2×256+tanh 而非现有 PyTorch 代码的 4×512+elu，原因：(a) 与 JAX 参考实现对齐，方便第八步中进行交叉验证；(b) JAX 版在简单环境上验证了该结构的有效性；(c) 如果 Go2 环境需要更大容量，可在后续实验中扩展为 4×512

- 网络初始化策略与 JAX 版对齐：隐藏层权重使用 `orthogonal(sqrt(2))`、偏置为 0；actor 最后一层使用 `orthogonal(0.01)`；critic 最后一层使用 `orthogonal(1.0)`

- 实现 `act(obs)` 方法：返回 `(action, log_prob, energy_value, reach_value)`
- 实现 `evaluate(obs)` 方法：返回 `(energy_value, reach_value)`
- 实现 `act_inference(obs)` 方法：确定性推理（用均值而非采样）
- `log_std` 作为可学习参数（`nn.Parameter`）

- 三个网络必须完全独立（不共享参数），接口设计要便于后续 EC-EFPPO 中分别计算梯度和分别创建优化器

### 工作重心

三网络的独立性和接口一致性。三个网络必须完全独立（不共享参数），但接口设计要便于后续 EC-EFPPO 中分别计算梯度。注意网络结构选择 2×256+tanh 是为了与 JAX 版对齐方便交叉验证，这是一个有意的设计决策。

---

## 第六步：实现 EC-EFPPO 算法核心

### 目标

实现完整的 EC-EFPPO 训练器，包括缓冲区、优势计算和三网络独立更新（含三个独立优化器）。

### 新建文件

`rsl_rl/rsl_rl/algorithms/ecfppo.py`

### 具体任务

**缓冲区：**
- 实现 `EC_EFPPO_Buffer` 类：在现有 `ReachAvoidBuffer` 的基础上扩展，额外存储 `energy` 序列、`energy_consumption` 序列和 `value_reach` 序列（reach critic 的预测值）。`g_values` 和 `h_values` 的存储保持不变
- 实现 `compute_advantages(last_values_energy, last_values_reach)` 方法，内部执行：(1) 调用 `calculate_indexs3` 计算 earliest reach index，得到 done 标志；(2) 将环境 `dones` OR 到 done 矩阵中；(3) 用 reach 序列和 reach value 序列调用 `_calculate_reach_gae` 得到 reach 优势和目标；(4) 用 energy value 序列和 reward 序列调用 `calculate_energy_gae` 得到 energy 优势和目标；(5) 构造组合信号 `g_append = max(reach, -energy)` 和 `V_total = max(V_reach, V_energy - energy)`，调用 `_calculate_reach_gae` 得到组合优势 `advantages_total`

**训练器：**
- 实现 `EC_EFPPO` 类。`__init__` 接收 `EC_EFPPO_ActorCritic` 实例和所有超参数。**创建三个独立的优化器**：`self.policy_optimizer = Adam(actor_critic.actor.parameters())`、`self.energy_optimizer = Adam(actor_critic.energy_critic.parameters())`、`self.reach_optimizer = Adam(actor_critic.reach_critic.parameters())`。这是与 JAX 版三个独立 `TrainState` 对应的设计——三个网络的梯度完全隔离，互不干扰
- 实现 `update()` 方法：多 epoch、多 mini-batch 的三路独立 PPO 更新。每次 mini-batch 更新分三步：(a) 用 `advantages_total` 计算 policy loss → `policy_optimizer.step()`；(b) 用 `targets_V` 计算 energy value loss → `energy_optimizer.step()`；(c) 用 `targets_h` 计算 reach value loss → `reach_optimizer.step()`。对应 JAX 版 `_ecefppo_update` 中三个独立的 loss function。注意：优势归一化在 policy loss 计算内部进行（`gae = (gae - gae.mean()) / (gae.std() + 1e-8)`），而非全局归一化
- 实现 γ 退火逻辑：`gamma_reach` 从 `gamma_reach_init` 到 `gamma_reach_final` 线性插值
- 实现 entropy 退火逻辑：`entropy_coef` 从初始值线性衰减到 0
- 实现梯度裁剪：三个优化器分别执行 `clip_grad_norm_(parameters, max_grad_norm)`

### 工作重心

三路优势计算的正确性和三网络更新的独立性。这是算法的数学核心，需要确保：`calculate_indexs3` 输出的 done 标志被正确传递到后续的三路 GAE 计算中；三个网络的梯度互不干扰（通过三个独立优化器实现）；Policy 的优势信号来自 combined 优势而非单一的 reach 或 energy 优势。

---

## 第七步：改造训练脚本

### 目标

编写适配 EC-EFPPO 的 Go2 训练循环，整合前几步的所有组件。

### 修改文件

- `legged_gym/scripts/train_reach_avoid.py`
- `legged_gym/envs/go2/go2_config.py`

### 具体任务

- 修改 `HierarchicalVecEnv` 类（定义在 `train_reach_avoid.py` 中），适配第三步中 `HierarchicalGO2Env` 新增的返回值：`reset()` 返回 `(obs, g_vals, h_vals, energy)`，`step()` 返回 `(obs, g_vals, h_vals, dones, infos, energy, energy_consumption)`
- 修改 `create_env` 函数，从配置中读取 EC-EFPPO 相关参数并传入环境
- 重写 `train_reach_avoid` 函数的核心训练循环：(1) 初始化 `EC_EFPPO_ActorCritic` 和 `EC_EFPPO` 实例；(2) rollout 阶段：收集 `(obs, g, h, energy, consumption, dones)` 到 buffer；(3) 优势计算阶段：调用 `buffer.compute_advantages()`；(4) 更新阶段：调用 `alg.update()`；(5) γ 和 entropy 退火
- 修改 `compute_reach_avoid_success_rate` 函数，增加能量消耗统计
- 在 `go2_config.py` 中新增 `GO2EC_EFPPOCfgPPO` 配置类，参数与 JAX 版 `arguments.py` 对齐：`gamma_energy=1.0`、`gamma_reach_init=0.999`、`gamma_reach_final=0.99999`、`entropy_coef=0.01`、`vf_coef=0.5`、`clip_eps=0.2`、`num_mini_batches=8`、`gae_lambda=0.95`、`max_grad_norm=0.5`、`anneal_ent=False`
- 保留现有的 checkpoint 保存和日志记录逻辑，日志中增加 `energy_loss`、`reach_loss`、`actor_loss`、`reach_gamma`、`entropy_weight` 等 EC-EFPPO 特有指标

### 工作重心

训练循环的完整性和日志的可追溯性。确保数据流从环境到 buffer 到算法层的每一步都不丢失，日志能清晰反映三路 loss 的变化趋势和 γ 退火的进度。

---

## 第八步：调试验证与性能对比

### 目标

在 Go2 环境上完成端到端训练，验证 EC-EFPPO 相比 Reach-Avoid PPO 的优势。

### 具体任务

- 小规模快速验证：用较少的并行环境（如 `num_envs=64`）和较短的训练轮数，确认三网络 loss 都在下降、energy value 能正确预测能量消耗趋势、reach value 能正确反映离目标的距离、earliest reach index 输出的 done 标志位置合理
- 全量训练：使用与基线相同的环境配置（`num_envs=4096`）进行完整训练，对比 EC-EFPPO 和 Reach-Avoid PPO 在成功率和能量消耗两个维度上的差异
- 与 JAX 版交叉验证：在简单环境（如 Pendulum）上同时运行 JAX 版和 PyTorch 版，对比训练曲线是否一致，作为算法移植正确性的最终验证
- 参数调优：重点关注 `gamma_reach_init/final` 的退火范围对训练稳定性的影响、`vf_coef` 对收敛速度的影响、energy 消耗缩放系数对策略能效的影响
- （可选改进）精确能量方案实验：从底层力矩计算消耗 `sum(torques²)`，与简单方案 `||actions||² × scale` 对比
- （可选改进）网络容量实验：尝试 4×512+elu 结构，对比 2×256+tanh 在 Go2 环境上的表现差异

### 工作重心

训练稳定性和算法效果的量化对比。EC-EFPPO 的核心价值在于"在保证到达目标的前提下最小化能量消耗"，需要通过对比实验证明这一点。
