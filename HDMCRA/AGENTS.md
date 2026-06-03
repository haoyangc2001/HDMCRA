# HDMCRA 项目指南

## 项目概述

HDMCRA（Hierarchical Distance-based Mini-Cost Reach-Avoid）是一个基于 IsaacGym 仿真的四足机器人导航训练项目。目标是将 Unitree Go2 的高层导航算法从 Reach-Avoid PPO 替换为 EC-EFPPO（Energy-Constrained Earliest Feasible PPO），使机器人在安全到达目标的同时最小化能量消耗。

### 当前阶段

**整体框架已搭建完成，进入训练验证阶段。** 实施计划的 15 个阶段中，阶段 1-7（基础搭建）和阶段 9-14（Bug 修复）已完成，阶段 8（全量训练与性能对比）待执行。

当前的核心任务是：**通过训练实验验证实现的正确性，分析训练数据，定位问题，确保算法正常收敛。**

具体包括：
- 每次训练后拿到训练数据（success rate、energy loss、reach loss、actor loss 等），分析数据是否正常
- 检查实现逻辑有没有偏差（与 JAX 参考对比、与基线对比）
- 诊断训练不收敛或不稳定的原因（梯度信号、超参数、环境交互等）
- 迭代修复，直到 EC-EFPPO 在 Go2 环境上稳定收敛

### 参考仓库

- **Go2HierarchicalReachAvoidRL**：PyTorch 基线（Reach-Avoid PPO），成功率 69-74%
- **Go2HierarchicalMiniCostReachAvoid**：JAX 参考实现（EC-EFPPO 原始算法）
- 详细计划见 `doc/plan/AchievePlan/plan.json`
- 调试记录见 `doc/plan/DebugPlan/debug_records.json`

## 开发环境

| 项目 | 规格 |
|---|---|
| Conda 环境名 | `hdmcr` |
| Python | 3.8.20（IsaacGym 要求 `>=3.6,<3.9`） |
| PyTorch | 1.13.1 + CUDA 11.7 |
| CUDA | 11.7（需与 PyTorch 编译版本一致） |
| IsaacGym | 1.0rc4（editable mode 安装） |
| GPU | NVIDIA GeForce RTX 4090 |
| conda 路径 | `/pub/data/caohy/miniconda/envs/hdmcr` |

**关键约束**：IsaacGym 对 Python、PyTorch、CUDA 版本有隐式要求，版本不匹配会导致 gymtorch 编译失败或运行时错误。

## 当前工作流程

### 训练验证循环

```
运行训练 → 收集数据 → 分析数据 → 定位问题 → 修复代码 → 重新训练
    ↑                                                    │
    └────────────────────────────────────────────────────┘
```

### 训练运行命令

```bash
# 设置环境变量
export LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH

# 小规模快速验证（推荐先跑这个确认代码无报错）
cd /home/caohy/repositories/HDMCRA/HDMCRA/legged_gym_go2
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 64 --max_iterations 50

# 全量训练
conda run -n hdmcr env LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  python legged_gym/scripts/train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500
```

### 数据分析要点

训练日志保存在 `legged_gym_go2/logs/ecfppo_go2/<timestamp>/training.log`，每行格式：

```
iter 00001 | success 0.000 | cost 0.0 | energy 0.0 | actor_loss -0.00563 | energy_loss 414.5 | reach_loss 92103.4 | entropy 4.2568 | gamma_reach 0.999000 | ent_coef 0.01000 | elapsed 14.55s
```

**需要关注的异常信号**：

| 指标 | 正常范围 | 异常信号 | 可能原因 |
|------|----------|----------|----------|
| `success` | 从 0 逐步上升 | 始终为 0 或剧烈波动 | g/h 信号不正确、超参数问题 |
| `energy_loss` | 稳定在 100~1000 | 爆炸到 10^10+ 后坍缩到 ≈0 | γ_energy 太大、energy targets 量级失控 |
| `reach_loss` | 稳定下降 | 不下降或爆炸 | g 函数信号不正确 |
| `actor_loss` | 在 0 附近波动 | 持续为 0 或 NaN | 梯度消失/爆炸、优势归一化问题 |
| `entropy` | 缓慢下降（如开启退火）| 骤降到 0 | 策略过早收敛、entropy 系数太小 |

**与基线对比**：基线 Reach-Avoid PPO 在相同环境下 50 轮即达 52% 成功率，1500 轮达 69%。EC-EFPPO 的目标是在保证成功率的前提下降低能量消耗。

## 已知问题与修复记录

详见 `doc/plan/DebugPlan/debug_records.json`，共 3 轮：

1. **Plan 1（2026-06-01）**：对齐基线超参数（网络 4×512+elu，LR 1e-3，vf_coef 1.0）— Peak 5.4%→40.8%
2. **Plan 2（2026-06-02）**：修复 γ_energy（1.0→0.99）— Peak→57.3%，但训练不稳定
3. **Plan 3（2026-06-03）**：修复 6 个实现正确性 Bug（success rate、reset 时序、能耗计算等）— 代码修复完成，待重新训练验证

### Plan 3 修复的 Bug 清单

| Bug | 严重性 | 问题 | 状态 |
|-----|--------|------|------|
| B1 | P0 | success rate 传入 energy 而非 h_values | ✅ 已修复 |
| B2 | P0 | buffer 没存 h_values | ✅ 已修复 |
| B3 | P0 | reset() 观测与能量不同步 | ✅ 已修复 |
| B4 | P1 | 能耗按未裁剪动作计算 | ✅ 已修复 |
| B5 | P1 | action_repeat 下能耗累计缺失 | ✅ 已修复 |
| B6 | P2 | energy 归一化方案错误 | ✅ 已修复 |

**重要**：Plan 1 和 Plan 2 的训练数据因 Bug B1-B5 存在，success rate 和能耗统计不可直接使用。修复后需要重新训练验证。

## 工作规范

### 决策点处理原则

实施过程中遇到任何需要做选择、但用户未明确指定的决策点时，**必须暂停当前工作，先向用户请示**，待获得明确指示后再继续执行。禁止自行假设或猜测用户的意图。

### 训练数据分析规范

每次训练完成后，**必须**分析以下内容并记录：

1. **Loss 趋势**：三路 loss（actor、energy、reach）是否正常下降/稳定？有无爆炸或坍缩？
2. **Success rate 趋势**：是否从 0 开始上升？上升速度是否合理？有无剧烈波动？
3. **Gamma reach 退火**：退火是否按预期进行？
4. **与基线对比**：相同迭代数下，EC-EFPPO 与基线 PPO 的成功率差距是多少？
5. **异常信号**：有无 NaN/Inf？有无 loss 爆炸？有无 success rate 突然归零？

**结合历史记录分析**：每次分析训练数据时，**必须**先读取 `doc/plan/DebugPlan/debug_records.json`，回顾之前的调试记录（Plan 1/2/3 的改动和结果），对比本次训练数据与历史数据的差异，判断问题是新出现的还是已知问题的延续。

**记录调试结果**：每次完成一轮 debug 优化（调参、修复 Bug、改动实现等）后，**必须**在 `doc/plan/DebugPlan/debug_records.json` 中新增一条记录，包含：
- `plan_id`：递增编号
- `plan_name`：本轮优化的简要名称
- `date`：日期
- `background`：为什么要做这个优化（基于什么数据/现象）
- `changes`：改动明细（文件、项目、旧值、新值、原因）
- `result`：实验结果（实验 ID、关键指标对比、发现）

分析结果同时记录到 `plan.json` 的 `completed_work` 中。

### 测试运行规范

修改代码后，**必须**运行相关测试确认无回归：

```bash
# 运行全部测试（42 个）
cd /home/caohy/repositories/HDMCRA/HDMCRA

# 不需要 isaacgym 的测试
conda run -n hdmcr python tests/test_ecfppo_gae.py
conda run -n hdmcr python tests/test_ecfppo.py
conda run -n hdmcr python tests/test_energy_state.py

# 需要 isaacgym 的测试
LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_train_ecfppo.py
LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH \
  conda run -n hdmcr python tests/test_ecfppo_actor_critic.py
```

### 任务完成后必须记录

每次完成一个步骤或子任务后，**必须**在 `plan.json` 对应步骤的 `completed_work` 字段中记录：
- 完成了哪些具体工作
- 遇到的问题及解决方案
- 修改了哪些文件
- 训练数据和分析结论

### 代码修改原则

- **不破坏现有功能**：每次修改前确认当前代码能正常运行，修改后立即验证
- **最小化改动范围**：只修改当前步骤涉及的文件，不做无关重构
- **对齐参考实现**：算法逻辑逐行对照 JAX 版本，确保数值行为一致
- **超参数偏离需记录**：如果决定偏离 JAX 默认值或原始计划，必须记录偏离原因和实际值

## 运行时注意事项

- **import 顺序**：必须先 `import isaacgym` 再 `import torch`，否则 CUDA 上下文冲突会导致进程挂起
- **环境变量**：运行前需设置 `LD_LIBRARY_PATH` 包含 conda env 的 lib 目录
- **测试路径**：测试文件中的 `sys.path.insert` 使用绝对路径，需与实际部署路径一致

```bash
# 推荐的运行前设置
export LD_LIBRARY_PATH=/pub/data/caohy/miniconda/envs/hdmcr/lib:$LD_LIBRARY_PATH
conda activate hdmcr
```
