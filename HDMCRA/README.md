# HDMCRA

**Hierarchical Distance-based Mini-Cost Reach-Avoid**

HDMCRA 是一个基于 IsaacGym 的 Unitree Go2 四足机器人导航训练项目。项目将高层导航算法从 Reach-Avoid PPO 替换为 **EC-EFPPO**（Energy-Constrained Earliest Feasible PPO），目标是在安全到达目标的同时考虑能量消耗。

本项目复用 Go2 基线中的低层运动控制策略，主要关注高层导航环境、EC-EFPPO 算法实现、训练循环和训练诊断。

## 当前状态

当前 EC-EFPPO 已经完成端到端训练所需的主体实现：

- 高层环境已经接入 energy 状态、energy consumption、`g/h` 值和 `high_level_action_repeat` 下的能耗累计。
- EC-EFPPO buffer、earliest-reach index、三路 GAE 和三个独立优化器已经实现。
- 回归测试覆盖 GAE 工具、buffer 行为、actor-critic 结构、energy 状态和训练脚本接口。

训练稳定性和最终效果仍在验证中。实现修复之前的训练结果只能作为历史调试数据，不能作为最终实验结论。当前项目阶段是分析训练日志、定位不稳定信号，并针对算法、环境或超参数做严谨的小步改动。

## 当前最新进展

截至 2026-06-09，训练稳定性诊断已经推进到 D007 阶段。当前重点不是继续盲目长训，而是解释 `success`（成功率）在短暂升高后为什么会断崖下降。

已经完成的关键修复和诊断包括：

- `std/log_std`（动作分布标准差）：已从直接优化 `std` 改为优化 `log_std`，并限制 `std` 上界，避免探索噪声无界增长。
- `reach_value_clip`（reach critic bootstrap 语义裁剪）：已限制 reach bootstrap value，避免极少量 open 样本把 target 拉到无语义量级。
- `energy_consumption_scale`（能耗缩放）：已把单个高层步最大能耗从约 120 降到约 8，缓解 energy 很快触底的问题。
- 三路学习率拆分：当前默认 `policy_learning_rate=1e-4`、`energy_learning_rate=1e-3`、`reach_learning_rate=3e-4`。
- `actor_mean_bound_coef=1e-2`（动作均值边界正则系数）：已明显降低 actor mean 发散、动作贴边、`energy_loss` 和 `reach_loss`。
- 最新诊断字段：已增加 `reach_rate`（到达目标比例）、`safe_rate`（安全比例）、`unsafe_before_reach`（到达前不安全比例）、`no_reach`（未到达比例）以及动作分量统计。

当前最新已完成短训 `20260609-121135` 表明：更强动作均值边界正则能稳定动作、能耗和 reach critic，但 `success` 仍会在高峰后坍塌。当前正在/应当运行 100-150 iter 诊断短训，重点判断失败主要来自 `no_reach`（未到达）、`unsafe_before_reach`（到达前不安全），还是某个动作维度异常。完整证据链见 `doc/debug.md`。

## 架构概览

本项目采用分层控制结构：

- **低层策略**：预训练 PPO 运动控制器，将速度命令转换为 Go2 关节动作。
- **高层策略**：EC-EFPPO 策略，输出 `[vx, vy, vyaw]` 速度命令。
- **环境信号**：高层观测包含目标方向、障碍物类 lidar 特征、机器人状态和 energy。
- **Reach-Avoid 信号**：`g` 表示目标到达条件，`h` 表示安全约束违反情况。
- **能量信号**：能耗由裁剪后的高层动作计算，并乘以 `high_level_action_repeat`。

EC-EFPPO 使用三个完全独立的网络：

- `actor`：输出高层速度命令的策略分布。
- `energy_critic`：预测 energy 相关 value target。
- `reach_critic`：预测 reach-avoid 相关 value target。

策略网络使用组合优势更新，energy critic 和 reach critic 分别使用各自的目标更新。

## 关键文件

| 路径 | 作用 |
|---|---|
| `legged_gym_go2/legged_gym/scripts/train_ecfppo.py` | EC-EFPPO 训练入口 |
| `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py` | 高层策略与低层运动策略的联合执行环境 |
| `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` | 高层观测、energy 状态和 `g/h` 函数 |
| `legged_gym_go2/legged_gym/envs/go2/go2_env.py` | Go2 IsaacGym 底层环境和安全/到达指标 |
| `legged_gym_go2/legged_gym/envs/go2/go2_config.py` | Go2、高层导航和 EC-EFPPO 配置 |
| `rsl_rl/rsl_rl/modules/actor_critic.py` | 基线 actor-critic 和 EC-EFPPO 三网络模块 |
| `rsl_rl/rsl_rl/algorithms/ecfppo.py` | EC-EFPPO buffer 和训练器 |
| `rsl_rl/rsl_rl/algorithms/ecfppo_gae.py` | earliest-reach index 和 GAE 工具函数 |
| `tests/` | 关键路径回归测试 |
| `doc/debug.md` | 当前训练诊断和调试记录 |

## 目录结构

```text
HDMCRA/
├── isaacgym/              # NVIDIA IsaacGym 包和资源
├── legged_gym_go2/        # Go2 环境、配置、脚本和资源
├── rsl_rl/                # 强化学习算法和模型模块
├── tests/                 # 回归测试
├── doc/                   # 当前调试记录
├── AGENTS.md              # 当前开发和训练诊断指南
└── README.md
```

## 环境要求

当前已知可用环境如下：

| 组件 | 版本 |
|---|---|
| Conda 环境 | `hdmcr` |
| Python | 3.8.20 |
| PyTorch | 1.13.1 + CUDA 11.7 |
| IsaacGym | 1.0rc4 |
| 开发使用 GPU | NVIDIA GeForce RTX 4090 |
| Conda 路径 | `/pub/data/caohy/miniconda/envs/hdmcr` |

IsaacGym 对 Python、PyTorch、CUDA 和 import 顺序比较敏感。运行测试或训练前需要设置 `LD_LIBRARY_PATH`，并确保先导入 `isaacgym` 再导入 `torch`。

```bash
export LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH
```

## 安装

从项目根目录开始：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA

cd isaacgym/python
pip install -e .

cd ../../rsl_rl
pip install -e .

cd ../legged_gym_go2
pip install -e .

pip install scipy opencv-python tensorboard pyyaml
```

验证导入：

```bash
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python -c "import isaacgym; import torch; import rsl_rl; import legged_gym; print(torch.__version__, torch.cuda.is_available())"
```

## 训练

建议先运行小规模验证：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2

conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 64 --max_iterations 50
```

运行完整 EC-EFPPO 训练：

```bash
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500
```

日志和 checkpoint 输出目录：

```text
legged_gym_go2/logs/ecfppo_go2/<timestamp>/
```

日志格式示例：

```text
iter 00010 | success 0.018 | cost 34.0 | energy 100.3 | reach_rate 0.021 | safe_rate 0.068 | unsafe_before_reach 0.003 | no_reach 0.979 | actor_loss 0.27018 | energy_loss 16.23113 | reach_loss 97329.93838 | mean_bound_loss 2.34582 | entropy 2.1555 | gamma_reach 0.999089 | ent_coef 0.00096 | elapsed 27.42s
debug 00010 | std_mean 0.4967 | e_cons [4.266e+00, 8.000e+00] | act_mean_clip_ratio 0.3266 | act_mean_abs_dim [3.889e+00, 4.507e-01, 1.021e-01] | act_mean_clip_dim [0.9784, 0.0013, 0.0000] | reach_clip_ratio 0.0000 | open_ratio 0.0310
```

字段说明：

| 字段 | 含义 |
|---|---|
| `success` | 成功率：安全到达目标的环境比例 |
| `cost` | 成功环境的平均首次到达时间步 |
| `energy` | 成功环境的平均能量消耗 |
| `reach_rate` | 到达目标比例，不考虑是否安全 |
| `safe_rate` | 到达前安全比例；未到达但全程安全也会计入 |
| `unsafe_before_reach` | 到达前已经不安全的比例 |
| `no_reach` | 未到达目标比例，是当前诊断 success 断崖的重点字段 |
| `actor_loss` | PPO 策略损失 |
| `energy_loss` | Energy critic 损失 |
| `reach_loss` | Reach critic 损失 |
| `mean_bound_loss` | actor mean 越过动作边界的正则损失 |
| `entropy` | 策略熵 |
| `gamma_reach` | 当前 reach 折扣因子 |
| `ent_coef` | 当前 entropy 系数 |
| `std_mean` | 动作分布标准差均值 |
| `e_cons` | 每步能量消耗的均值和最大值 |
| `act_mean_clip_ratio` | actor mean 超出 `[-1, 1]` 的比例 |
| `act_mean_abs_dim` | 各动作维度的 actor mean 绝对值均值，三维通常对应 `[vx, vy, vyaw]` |
| `act_mean_clip_dim` | 各动作维度 actor mean 越界比例，定位是否某个动作维度主导异常 |
| `reach_clip_ratio` | reach critic bootstrap value 被语义裁剪的比例 |
| `open_ratio` | reach GAE 中 non-done/open 样本比例 |

## 测试

运行不依赖 IsaacGym 的轻量测试：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA

conda run -n hdmcr python tests/test_ecfppo_gae.py
conda run -n hdmcr python tests/test_ecfppo.py
conda run -n hdmcr python tests/test_energy_state.py
```

运行需要导入 IsaacGym 的测试：

```bash
LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_train_ecfppo.py

LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_ecfppo_actor_critic.py
```

## 训练诊断重点

分析训练结果时重点关注：

- `success`（成功率）是否上升并保持稳定，而不是震荡或坍缩。
- `reach_rate`（到达目标比例）和 `no_reach`（未到达比例）：如果 `no_reach` 长期接近 1，优先判断策略是否学不到目标驱动或动作被正则压得过小。
- `safe_rate`（安全比例）和 `unsafe_before_reach`（到达前不安全比例）：如果到达比例提高但不安全比例也提高，优先检查避障/安全约束。
- `act_mean_clip_dim`（各动作维度均值越界比例）：如果某一维明显更高，优先检查该维动作语义、归一化和代价尺度。
- `energy_loss` 和 `reach_loss` 是否处于可学习的量级。
- `reach_clip_ratio` 是否长期很高；如果很高，说明 reach critic 仍大量越过语义边界。
- `actor_loss` 是否为有限值，并能反映策略更新。
- `entropy` 是否符合当前 entropy 设置和退火策略。
- `gamma_reach` 是否按配置退火到预期值。
- 在相同环境配置下，EC-EFPPO 与 Reach-Avoid PPO 基线的差距在哪里。

当前目标不是单纯跑完训练，而是判断哪些信号或实现选择导致训练无法稳定收敛。接手项目时应先读 `doc/debug.md` 的“新手接手摘要”和 D007 记录，再决定下一轮训练或代码改动。

## 参考

- `Go2HierarchicalReachAvoidRL/`：PyTorch Reach-Avoid PPO 基线。
- `Go2HierarchicalMiniCostReachAvoid/`：JAX EC-EFPPO 参考实现。
- `doc/debug.md`：当前训练诊断和调试记录。
- `AGENTS.md`：当前开发流程和训练诊断规则。
