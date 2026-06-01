# HDMCRA

**Hierarchical Distance-based Mini-Cost Reach-Avoid**

基于 IsaacGym 仿真的 Unitree Go2 四足机器人能量约束导航训练项目。

本项目将 Go2 的高层导航算法从 Reach-Avoid PPO 替换为 **EC-EFPPO**（Energy-Constrained Earliest Feasible PPO），使机器人在复杂环境中安全到达目标的同时**最小化能量消耗**。

---

## 项目背景

在四足机器人的导航任务中，传统的 Reach-Avoid 强化学习算法只关注"能否安全到达目标"，而忽略了能量消耗。EC-EFPPO 在此基础上引入了能量约束机制，通过以下核心改进实现能效优化：

- **三网络独立架构**：Policy（策略）、Energy Value（能量价值）、Reach Value（可达价值）三个网络完全独立，分别优化
- **Earliest Reach Index 算法**：反向扫描轨迹，找到使组合价值最小的时间步作为 done 标志
- **三路 GAE 优势估计**：分别计算 energy 优势、reach 优势和组合优势，为三个网络提供精确的训练信号
- **γ 退火机制**：reach 折扣因子从 `gamma_reach_init` 线性增长到 `gamma_reach_final`，平衡训练稳定性与收敛速度

## 项目结构

```
HDMCRA/
├── isaacgym/                        # NVIDIA IsaacGym 物理仿真引擎（预编译）
├── legged_gym_go2/                  # Go2 机器人环境与训练脚本
│   ├── legged_gym/
│   │   ├── envs/
│   │   │   ├── go2/
│   │   │   │   ├── go2_config.py            # 环境与算法配置
│   │   │   │   ├── go2_env.py               # 底层 IsaacGym 环境
│   │   │   │   ├── hierarchical_go2_env.py   # 分层环境（透传 energy 数据）
│   │   │   │   └── high_level_navigation_env.py  # 高层导航环境（energy 状态）
│   │   │   └── base/                        # 基础环境类
│   │   ├── scripts/
│   │   │   ├── train_ecfppo.py              # EC-EFPPO 训练脚本 ★
│   │   │   ├── train_reach_avoid.py         # 基线 Reach-Avoid PPO 训练脚本
│   │   │   ├── play_reach_avoid.py          # 可视化评估脚本
│   │   │   └── test_reach_avoid.py          # 测试脚本
│   │   └── utils/                           # 工具函数
│   └── resources/robots/go2/                # Go2 URDF 模型
├── rsl_rl/                          # 强化学习算法库
│   ├── rsl_rl/
│   │   ├── modules/
│   │   │   ├── actor_critic.py              # ActorCritic + EC_EFPPO_ActorCritic ★
│   │   │   └── actor_critic_recurrent.py
│   │   ├── algorithms/
│   │   │   ├── ecfppo.py                    # EC-EFPPO 训练器 + Buffer ★
│   │   │   ├── ecfppo_gae.py                # 三路 GAE 算法 ★
│   │   │   ├── ppo.py                       # 标准 PPO
│   │   │   └── reach_avoid_ppo.py           # Reach-Avoid PPO 基线
│   │   └── storage/
│   └── setup.py
├── tests/                           # 回归测试
│   ├── test_ecfppo_actor_critic.py          # 三网络架构测试（12 个）
│   ├── test_ecfppo_gae.py                   # GAE 算法测试（9 个）
│   ├── test_ecfppo.py                       # EC-EFPPO 训练器测试（13 个）
│   ├── test_energy_state.py                 # Energy 状态测试（8 个）
│   └── test_train_ecfppo.py                 # 训练脚本测试（9 个）
├── doc/plan/                        # 开发计划
│   ├── plan.json                            # 详细实施计划（8 个阶段）
│   └── plan.md                              # 计划文档
├── setup.py                         # 包安装配置
└── AGENTS.md                        # 开发规范
```

★ 标记为本项目新增的核心文件。

## 算法架构

```
                    ┌─────────────────────────────────────────┐
                    │           EC_EFPPO_ActorCritic          │
                    │                                         │
  obs ──────────────┤  ┌──────────┐  ┌───────────────┐       │
                    │  │  Actor   │  │ Energy Critic │       │
                    │  │ 2×256    │  │ 2×256 + tanh  │       │
                    │  │ + tanh   │  │               │       │
                    │  └────┬─────┘  └──────┬────────┘       │
                    │       │               │                 │
                    │       ▼               ▼                 │
                    │   action dist    energy_value           │
                    │                                         │
                    │  ┌───────────────┐                      │
                    │  │ Reach Critic  │                      │
                    │  │ 2×256 + tanh  │                      │
                    │  └──────┬────────┘                      │
                    │         ▼                               │
                    │    reach_value                          │
                    └─────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            advantages_total    targets_energy     targets_reach
            (组合优势)         (energy 目标)      (reach 目标)
                    │                 │                 │
                    ▼                 ▼                 ▼
              Policy 更新      Energy 更新        Reach 更新
            (optimizer 1)    (optimizer 2)      (optimizer 3)
```

### 核心数据流

1. **Rollout 阶段**：环境返回 `(obs, g_vals, h_vals, energy, energy_consumption)`
2. **优势计算**：
   - `calculate_indexs3` → earliest reach index + done 矩阵
   - `calculate_reach_gae` → reach 优势 (γ_reach 退火)
   - `calculate_energy_gae` → energy 优势 (γ_energy=1.0)
   - 组合信号 `g_append = max(reach, -energy)` → 组合优势
3. **三路独立更新**：Policy 用组合优势，两个 Critic 各用各自的目标

## 快速上手

### 环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 20.04/22.04 | IsaacGym 仅支持 Linux |
| GPU | NVIDIA GPU（≥8GB 显存） | 必须支持 CUDA |
| CUDA | 11.7+ | 需与 PyTorch 编译版本一致 |
| Python | 3.8（`>=3.6, <3.9`） | IsaacGym 硬性要求 |
| PyTorch | 1.13.1 + CUDA 11.7 | 需与 gymtorch C++ 扩展兼容 |
| Conda | Miniconda/Anaconda | 用于创建隔离环境 |

### 第一步：创建 Conda 环境

```bash
# 创建 Python 3.8 环境
conda create -n hdmcr python=3.8 -y
conda activate hdmcr

# 安装 PyTorch 1.13.1 + CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：如果清华源下载失败，去掉 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 使用默认源。

### 第二步：安装项目依赖

```bash
# 进入项目目录
cd /path/to/HDMCRA/HDMCRA

# 安装 IsaacGym（物理仿真引擎）
cd isaacgym/python
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ../..

# 安装 rsl_rl（强化学习算法库）
cd rsl_rl
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# 安装 legged_gym_go2（Go2 机器人环境）
cd legged_gym_go2
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# 安装可选依赖
pip install scipy opencv-python tensorboard pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第三步：验证安装

```bash
# 设置环境变量（重要！）
export LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH

# 验证 import（必须按顺序：先 isaacgym 再 torch）
python -c "import isaacgym; import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('OK')"
```

如果输出 `OK` 且 `CUDA: True`，说明安装成功。

> **常见问题**：如果出现 `ImportError: libpython3.8.so.1.0: cannot open shared object file`，说明 `LD_LIBRARY_PATH` 未正确设置，请重新执行 `export LD_LIBRARY_PATH=...`。

### 第四步：运行训练

#### EC-EFPPO 训练（本项目核心）

```bash
# 进入 legged_gym_go2 目录
cd legged_gym_go2

# 小规模快速验证（推荐先跑这个确认环境正常）
conda run -n hdmcr env LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 64 --max_iterations 10

# 正式训练（num_envs=4096，约 1500 轮）
conda run -n hdmcr env LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500
```

训练日志保存在 `legged_gym_go2/logs/ecfppo_go2/<timestamp>/` 目录下：
- `training.log`：训练指标日志（success, cost, energy, losses, gamma_reach, ...）
- `model_<iter>.pt`：周期性 checkpoint
- `model_final.pt`：最终模型

#### 基线训练（Reach-Avoid PPO，用于对比）

```bash
conda run -n hdmcr env LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_reach_avoid.py --headless --num_envs 4096 --max_iterations 1500
```

### 第五步：查看训练结果

```bash
# 查看训练日志
tail -f legged_gym_go2/logs/ecfppo_go2/<timestamp>/training.log

# 日志格式示例：
# iter 00001 | success 0.000 | cost 0.0 | energy 0.0 | actor_loss -0.00563 | energy_loss 414.5 | reach_loss 92103.4 | entropy 4.2568 | gamma_reach 0.999000 | ent_coef 0.01000 | elapsed 14.55s
```

日志字段说明：

| 字段 | 含义 |
|------|------|
| `success` | 成功率（安全到达目标的环境比例） |
| `cost` | 成功环境的平均到达时间步 |
| `energy` | 成功环境的平均能量消耗 |
| `actor_loss` | 策略损失（PPO clip 目标） |
| `energy_loss` | Energy Value 网络损失 |
| `reach_loss` | Reach Value 网络损失 |
| `entropy` | 策略熵（探索程度） |
| `gamma_reach` | 当前 reach 折扣因子（退火中） |
| `ent_coef` | 当前 entropy 系数 |

### 第六步：可视化评估

```bash
# 使用训练好的模型进行可视化（需要显示器或虚拟显示）
python legged_gym/scripts/play_reach_avoid.py --load_run <timestamp>
```

### 运行测试

```bash
# 运行全部回归测试（51 个测试）
cd /path/to/HDMCRA/HDMCRA

# 测试三网络架构（12 个）
conda run -n hdmcr python tests/test_ecfppo_actor_critic.py

# 测试 GAE 算法（9 个）
conda run -n hdmcr python tests/test_ecfppo_gae.py

# 测试 EC-EFPPO 训练器（13 个）
conda run -n hdmcr python tests/test_ecfppo.py

# 测试 Energy 状态（8 个）
conda run -n hdmcr python tests/test_energy_state.py

# 测试训练脚本集成（9 个，需要 isaacgym）
conda run -n hdmcr env LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python tests/test_train_ecfppo.py
```

## 训练参数配置

EC-EFPPO 的超参数在 `legged_gym_go2/legged_gym/envs/go2/go2_config.py` 中的 `GO2EC_EFPPOCfgPPO` 类定义，与 JAX 参考实现对齐：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma_energy` | 1.0 | Energy 折扣因子（无折扣） |
| `gamma_reach_init` | 0.999 | Reach 折扣因子初始值 |
| `gamma_reach_final` | 0.99999 | Reach 折扣因子终止值 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_eps` | 0.2 | PPO clip 范围 |
| `vf_coef` | 0.5 | Value function loss 系数 |
| `entropy_coef` | 0.01 | Entropy 系数 |
| `max_grad_norm` | 0.5 | 梯度裁剪范数 |
| `learning_rate` | 3e-4 | 学习率 |
| `num_learning_epochs` | 10 | 每次更新的训练轮数 |
| `num_mini_batches` | 8 | Mini-batch 数量 |
| `num_steps_per_env` | 200 | 每轮 rollout 的时间步数 |
| `max_iterations` | 1500 | 最大训练轮数 |
| `hidden_dim` | 256 | 隐藏层维度（2 层） |
| `activation` | tanh | 激活函数 |

可以通过命令行覆盖配置：

```bash
python legged_gym/scripts/train_ecfppo.py --headless --num_envs 4096 --max_iterations 2000
```

## 核心模块说明

### `EC_EFPPO_ActorCritic`（三网络架构）

```python
from rsl_rl.modules.actor_critic import EC_EFPPO_ActorCritic

model = EC_EFPPO_ActorCritic(
    num_actor_obs=41,      # actor 观测维度（含 energy 状态）
    num_critic_obs=41,     # critic 观测维度
    num_actions=3,         # 动作维度 [vx, vy, vyaw]
    hidden_dim=256,        # 隐藏层维度
    num_hidden_layers=2,   # 隐藏层数
)

# 采样动作 + 计算 value
action, log_prob, energy_value, reach_value = model.act(obs, critic_obs)

# 仅计算 value（用于 bootstrap）
energy_value, reach_value = model.evaluate(critic_obs)

# 确定性推理（部署用）
action_mean = model.act_inference(obs)
```

### `EC_EFPPO_Buffer`（经验缓冲区）

```python
from rsl_rl.algorithms.ecfppo import EC_EFPPO_Buffer

buffer = EC_EFPPO_Buffer(num_envs=64, horizon=200, obs_shape=(41,), action_shape=(3,), device=device)

# 存储 transition
buffer.add(obs, actions, log_probs, values, value_reach,
           energy, energy_consumption, g_values, dones,
           next_obs, next_energy, next_g)

# 计算三路优势
buffer.compute_advantages(last_energy, last_reach,
                          gamma_energy=1.0, gamma_reach=0.99999,
                          gae_lambda=0.95, gamma_reach_init=0.999)
```

### `EC_EFPPO`（训练器）

```python
from rsl_rl.algorithms.ecfppo import EC_EFPPO

alg = EC_EFPPO(actor_critic=model, learning_rate=3e-4, ...)
alg.init_storage(num_envs=64, horizon=200, obs_shape=(41,), action_shape=(3,))

# 采样
actions, log_probs, vals_e, vals_r = alg.act(obs)

# 三路独立更新
loss_dict = alg.update(gamma_reach=0.99999, entropy_coef=0.01)
# loss_dict: {"actor_loss", "energy_loss", "reach_loss", "entropy_loss"}
```

## 技术参考

- **JAX 参考实现**：`Go2HierarchicalMiniCostReachAvoid/`（原始 EC-EFPPO 算法）
- **PyTorch 基线**：`Go2HierarchicalReachAvoidRL/`（Reach-Avoid PPO 基线）
- **IsaacGym**：NVIDIA GPU 物理仿真引擎
- **rsl_rl**：ETH Zurich 的强化学习库（本项目扩展了其中的 modules 和 algorithms）

## 开发阶段

本项目分 8 个阶段完成，详细实施记录见 `doc/plan/plan.json`：

| 阶段 | 名称 | 状态 |
|------|------|------|
| 1 | 搭建项目骨架与开发环境 | ✅ |
| 2 | 环境层改造 — 加入 Energy 状态 | ✅ |
| 3 | 分层环境改造 — 透传 Energy 数据 | ✅ |
| 4 | 移植 Earliest Reach Index 和三路 GAE | ✅ |
| 5 | 实现三网络架构 | ✅ |
| 6 | 实现 EC-EFPPO 算法核心 | ✅ |
| 7 | 改造训练脚本 | ✅ |
| 8 | 调试验证与性能对比 | ✅ |

## 常见问题

### Q: 出现 `ImportError: PyTorch was imported before isaacgym modules`

**原因**：IsaacGym 必须在 PyTorch 之前导入。

**解决**：确保代码中 `import isaacgym` 在 `import torch` 之前。

### Q: 出现 `ImportError: libpython3.8.so.1.0: cannot open shared object file`

**原因**：`LD_LIBRARY_PATH` 未包含 conda 环境的 lib 目录。

**解决**：
```bash
export LD_LIBRARY_PATH=$(conda info --base)/envs/hdmcr/lib:$LD_LIBRARY_PATH
```

### Q: gymtorch C++ 扩展编译失败

**原因**：PyTorch 版本与 IsaacGym 不兼容。

**解决**：确保使用 PyTorch 1.13.1 + CUDA 11.7。如果仍然失败，检查系统 CUDA 版本是否与 PyTorch 编译版本一致。

### Q: 训练时 GPU 内存不足

**解决**：减少 `num_envs`，如 `--num_envs 1024`。RTX 4090 (24GB) 可支持 `num_envs=4096`。

### Q: 如何恢复中断的训练

```bash
python legged_gym/scripts/train_ecfppo.py --headless --resume
```

在 `go2_config.py` 中设置 `resume_path` 指向 checkpoint 文件。
