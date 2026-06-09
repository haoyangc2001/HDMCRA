# HDMCRA 项目指南

本文档描述 HDMCRA 当前阶段的开发方式，面向后续开发会话和接手本仓库的编码代理。

## 当前阶段

HDMCRA 已经完成 EC-EFPPO 的主体工程实现。当前工作重点是 **训练稳定性诊断**。

已经完成的实现包括：

- Go2 高层导航环境已经接入 energy 状态和 energy consumption。
- `g_values`、`h_values`、energy 和 energy consumption 已经写入 EC-EFPPO buffer。
- earliest-reach index、energy GAE、reach GAE 和 combined advantage 已经实现。
- Policy、energy critic 和 reach critic 使用独立网络与独立优化器。
- success rate 统计、reset 时序、动作裁剪、action-repeat 能耗累计等 P0/P1 实现问题已经修复。

当前未完成的问题：

- 训练尚未证明可以稳定收敛到 Reach-Avoid PPO 基线水平。
- 后续需要通过训练日志判断问题来自算法逻辑、超参数、环境信号、归一化方案，还是实现细节。

不要把项目描述成“已完成验证”。当前更准确的描述是：**实现已完成，训练稳定性仍在验证中**。

## 当前最新诊断焦点

截至 2026-06-09，当前诊断主线是 D007：`actor mean`（策略动作均值）发散导致 `raw action`（原始采样动作）与 `clipped action`（环境实际执行动作）语义错配。

已经确认的事实：

- `log_std`（动作标准差的对数参数）已经受控，`std`（动作标准差）不再无界增长。
- `reach_value_clip=5000.0` 已经抑制极端 reach bootstrap target，但不能单独解决 success 坍塌。
- `energy_consumption_scale=8.0 / (num_actions * high_level_action_repeat)` 已经把单步最大高层能耗压到约 8，能量触底不再发生在最初 5-6 个高层步。
- `policy_learning_rate=1e-4` 和 `reach_learning_rate=3e-4` 比最初统一 `1e-3` 更稳定。
- `actor_mean_bound_coef=1e-2` 明显降低 `act_mean_clip_ratio`（动作均值越界比例）、`energy_loss`（能量价值损失）和 `reach_loss`（到达/避障价值损失）。
- 但 `success`（成功率）仍会在短暂高峰后断崖下降，所以当前不要继续盲目加大正则或继续长训。

当前下一步：跑 100-150 iter 诊断短训，分析 `reach_rate`（到达目标比例）、`safe_rate`（安全比例）、`unsafe_before_reach`（到达前不安全比例）、`no_reach`（未到达比例）和 `act_mean_clip_dim`（各动作维度均值越界比例）。

如果接手时已经有新日志，优先判断：

- `no_reach` 高：策略主要是不去目标，优先检查目标驱动、动作幅度是否被压小、advantage 是否给出有效到达信号。
- `unsafe_before_reach` 高：策略能接近目标但安全失败，优先检查避障约束、`h` 值定义和动作方向。
- 某个 `act_mean_clip_dim` 显著高：优先检查该动作维度的语义、归一化、速度范围和能耗/安全代价。
- `reach_clip_ratio` 高：reach critic 输出仍越过语义边界，优先检查 reach critic 学习率、梯度裁剪或输出约束。

## 重要原则

当前代码实现不能默认视为完全正确、严谨或设计合理。训练稳定性诊断阶段允许修改实现和设计，但必须基于严谨分析。

可以改动的范围包括：

- 算法语义，例如 advantage 构造、done mask、bootstrap、value target 定义。
- 环境信号，例如 `g/h` 定义、energy 计算、观测归一化、action repeat 语义。
- 训练超参数，例如 learning rate、`gamma_energy`、`gamma_reach`、`vf_coef`、entropy 系数。
- 网络结构和初始化策略。
- 日志统计口径和评价指标。

改动前必须说明：

- 当前训练日志暴露了什么现象。
- 现象可能对应哪条代码链路或数学定义。
- 为什么当前设计可能不合理。
- 准备如何验证改动是否有效。
- 哪些测试或对比实验可以防止引入回归。

禁止只凭感觉大改。每次改动都应该服务于一个明确、可验证的假设。

## 当前工作假设

- 当前实现是基于 JAX EC-EFPPO 参考实现、面向 Go2 任务的 PyTorch 适配版，不是逐行等价且已完成交叉验证的复刻版。
- 当前 Go2 EC-EFPPO 默认网络是 `4x512 + elu`，不是早期设计中的 `2x256 + tanh`。
- 当前 `gamma_energy` 是 `0.99`，不是原始默认值 `1.0`。
- 当前 `log_std_max=-0.6931471805599453`，对应最大 `std≈0.5`。
- 当前 `actor_mean_bound=1.0`、`actor_mean_bound_coef=1e-2`，用于惩罚 actor mean 超出动作边界。
- 当前三路学习率为 `policy_learning_rate=1e-4`、`energy_learning_rate=1e-3`、`reach_learning_rate=3e-4`。
- 当前 `reach_value_clip=5000.0`，用于限制 reach bootstrap value 的语义范围。
- 当前 `debug_stats_interval=10`，训练日志每 10 iter 输出分组 debug 字段。
- P0/P1 修复前的 success rate 和 energy consumption 统计只能作为历史调试参考。
- 新的训练结论必须基于修复后的代码和最新训练日志。

## 主要文件

| 路径 | 作用 |
|---|---|
| `legged_gym_go2/legged_gym/scripts/train_ecfppo.py` | EC-EFPPO 主训练循环 |
| `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` | 高层观测、energy 和 `g/h` 逻辑 |
| `legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py` | 高层动作通过低层策略执行 |
| `legged_gym_go2/legged_gym/envs/go2/go2_config.py` | 训练、网络和环境配置 |
| `rsl_rl/rsl_rl/algorithms/ecfppo.py` | Buffer、优势计算和三路更新 |
| `rsl_rl/rsl_rl/algorithms/ecfppo_gae.py` | earliest-reach index 和 GAE 工具 |
| `rsl_rl/rsl_rl/modules/actor_critic.py` | EC-EFPPO actor、energy critic 和 reach critic |
| `tests/` | 回归测试 |
| `doc/debug.md` | 当前训练诊断和调试记录 |

## 运行环境

当前已知可用环境：

| 项目 | 值 |
|---|---|
| Conda 环境 | `hdmcr` |
| Python | 3.8.20 |
| PyTorch | 1.13.1 + CUDA 11.7 |
| IsaacGym | 1.0rc4 |
| Conda 路径 | `/pub/data/caohy/miniconda/envs/hdmcr` |

运行测试或训练前设置：

```bash
export LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH
```

import 顺序很重要：必须先导入 `isaacgym`，再导入 `torch`。

## 训练命令

小规模冒烟测试：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 64 --max_iterations 50
```

完整 EC-EFPPO 训练：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500
```

训练日志位置：

```text
legged_gym_go2/logs/ecfppo_go2/<timestamp>/training.log
```

## 训练分析清单

每次训练完成或中断后，需要检查：

- `success`（成功率）：趋势、峰值、最终值，是否震荡、坍缩或长期不升。
- `reach_rate`（到达目标比例）：判断策略是否具备到达能力。
- `safe_rate`（安全比例）：判断安全约束是否主要瓶颈。
- `unsafe_before_reach`（到达前不安全比例）：判断失败是否来自提前碰撞/进入不安全区域。
- `no_reach`（未到达比例）：判断失败是否主要来自不去目标。
- `cost`（平均首次到达时间步）：成功轨迹是否更快到达，还是只出现随机成功。
- `energy`（成功平均能耗）：成功轨迹是否真的更省能，还是只反映初始状态更容易。
- `act_mean_clip_ratio`（动作均值越界比例）和 `act_mean_clip_dim`（各动作维度均值越界比例）：判断 actor mean 是否继续跑出动作边界，以及是否集中在某个维度。
- `reach_clip_ratio`（reach value 裁剪比例）：判断 reach critic 是否仍大量越过语义边界。
- `actor_loss`：是否有限，是否存在有效策略更新。
- `energy_loss`：是否爆炸、坍缩，或处于稳定可学习量级。
- `reach_loss`：是否爆炸、坍缩，或处于稳定可学习量级。
- `entropy`：是否符合当前探索策略和 entropy 设置。
- `gamma_reach`：是否按配置正确退火。
- 是否出现 NaN、Inf、指标突跳或 success rate 突然归零。

与 Reach-Avoid PPO 基线对比时，必须确认环境配置、训练轮数和评价窗口具有可比性。

## 日志解释规则

后续解释训练日志、debug 字段或代码变量时，首次出现的英文变量名必须尽量用括号补充中文含义，方便理解和复盘。

推荐写法：

- `success`（成功率）：当前评估窗口内成功到达目标的比例。
- `reach_loss`（到达/避障价值损失）：reach critic 的训练误差。
- `act_clip_ratio`（动作贴边比例）：动作被裁剪到 `[-1, 1]` 边界附近的比例。

如果变量含义不确定，必须明确写成“推测含义”，不要把不确定解释写成结论。

## 诊断方法

推荐按以下顺序定位问题：

1. 先读最新 `training.log`，确认异常最先出现在哪个指标。
2. 回到数据链路检查：环境返回值、buffer 存储、advantage 计算、loss 计算和日志统计是否时间对齐。
3. 对照测试覆盖范围，判断当前问题是否已有测试保护。
4. 如果怀疑算法语义问题，先构造小张量测试或简化环境验证，再改 Go2 全量训练。
5. 如果改超参数，先做小规模训练验证，再做长训练。
6. 如果改环境信号或算法定义，必须补充测试并记录设计理由。

## 调试记录规则

分析新训练结果前，先阅读：

```text
doc/debug.md
```

当一次训练结果导致新的判断或改动时，需要记录。以下情况应该新增调试记录：

- 修改超参数。
- 修改算法逻辑。
- 修改环境信号。
- 修改归一化方案。
- 修复实现 Bug。
- 得到关键训练结论。

每条新记录应包含：

- `ID`：例如 `D001`、`D002`。
- `状态`：open / resolved / abandoned。
- `严重性`：P0 / P1 / P2。
- `触发原因`：由什么日志、测试或代码现象触发。
- `现象`、`假设`、`代码链路`、`证据`、`结论`。
- `改动`：如果修改了代码或配置，记录文件、旧值、新值和原因。
- `验证`：记录测试、训练命令和结果。
- `后续动作`：下一步要做什么。

`doc/debug.md` 从当前训练稳定性诊断阶段重新开始记录。旧实现计划和旧 debug JSON 已清理，不再作为当前判断依据。

## 测试规则

代码改动后至少运行相关测试。基础测试：

```bash
cd /home/caohy/repositories/HDMCRA/HDMCRA
conda run -n hdmcr python tests/test_ecfppo_gae.py
conda run -n hdmcr python tests/test_ecfppo.py
conda run -n hdmcr python tests/test_energy_state.py
```

修改训练脚本、actor-critic 模块或环境链路时，运行 IsaacGym 相关测试：

```bash
LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_train_ecfppo.py

LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_ecfppo_actor_critic.py
```

如果因为 IsaacGym、CUDA 或环境限制无法运行某个测试，最终说明必须明确写出。

## 改动原则

- 每次改动都要对应一个从训练数据或代码审查中得到的明确假设。
- 不要把无关重构混入训练诊断改动。
- 保持数据流语义清晰：`obs`、`g_values`、`h_values`、`energy`、`energy_consumption` 和 `dones` 必须时间对齐。
- success 和 energy consumption 统计必须保持 `g/h` post-step 序列与 energy 序列的对齐关系。
- 超参数改动视为实验，必须记录旧值、新值和原因。
- 算法语义改动属于高风险改动，必须同步测试和文档中的语义说明。

## 新手接手步骤

如果第一次接手本仓库，按以下顺序建立上下文：

1. 先读 `README.md` 的“当前最新进展”和“训练诊断重点”，理解项目不是已完成收敛验证，而是在训练稳定性诊断阶段。
2. 再读 `doc/debug.md` 的“新手接手摘要”“当前待分析问题”和 D007 记录，确认最近一次已经验证了什么、下一步为什么要看 success 分解。
3. 打开最新 `legged_gym_go2/logs/ecfppo_go2/<timestamp>/training.log`，优先解析 `success/reach_rate/safe_rate/unsafe_before_reach/no_reach` 和 debug 行的动作分量。
4. 如果要改代码，先在 `doc/debug.md` 写清楚假设和验证计划，再做最小改动。
5. 如果只是继续训练，优先 100-150 iter 诊断短训，不要直接从不稳定 checkpoint 长训 1500 iter。

## 文档规则

- `README.md` 用于项目介绍、架构、环境、训练命令和高层状态说明。
- `AGENTS.md` 用于当前开发流程、训练诊断规则和改动原则。
- `doc/debug.md` 用于当前训练诊断、问题分析、实验结果和设计决策记录。
- 当前 `doc` 目录只保留 `debug.md` 作为调试入口。历史计划和旧 debug 记录已经清理，后续分析从当前实现和最新训练日志重新开始。
