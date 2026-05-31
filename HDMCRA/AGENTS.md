# HDMCRA 项目指南

## 项目概述

HDMCRA（Hierarchical Distance-based Mini-Cost Reach-Avoid）是一个基于 IsaacGym 仿真的四足机器人导航训练项目。目标是将 Unitree Go2 的高层导航算法从 Reach-Avoid PPO 替换为 EC-EFPPO（Energy-Constrained Earliest Feasible PPO），使机器人在安全到达目标的同时最小化能量消耗。

核心改动集中在环境层（加入 energy 状态）和算法层（三网络架构 + earliest reach index），底层 IsaacGym 仿真和低层运动策略完全复用现有代码。

详细实施计划见 [plan.json](/home/tanshan/rep/HDMCRA/HDMCRA/doc/plan/plan.json)。

## 源码参考

- **Go2HierarchicalReachAvoidRL**：现有 Reach-Avoid PPO 实现（PyTorch），包含 `legged_gym_go2/`、`rsl_rl/`、`isaacgym/`
- **Go2HierarchicalMiniCostReachAvoid**：EC-EFPPO 的 JAX 参考实现，包含 `rl/gae.py`（GAE 算法）、`rl/EC-EFPPO.py`（训练循环）、`model/actorcritic.py`（网络结构）

## 开发环境

| 项目 | 规格 |
|---|---|
| Conda 环境名 | `hdmcr` |
| Python | 3.8（IsaacGym 要求 `>=3.6,<3.9`） |
| PyTorch | 需与 IsaacGym 兼容，参考 `rlgpu_conda_env.yml`（pytorch=1.8.1 + cudatoolkit=11.1），可尝试 1.12+ 但需确认 gymtorch 编译通过 |
| CUDA | 需与 PyTorch 编译版本一致 |
| IsaacGym | 预编译二进制，不可修改，通过 `pip install -e .` 安装 |
| GPU | 必须有 NVIDIA GPU，`nvidia-smi` 需正常输出 |

**关键约束**：IsaacGym 对 Python、PyTorch、CUDA 版本有隐式要求，版本不匹配会导致 gymtorch 编译失败或运行时错误。安装时严格按照 `rlgpu_conda_env.yml` 的版本，遇到问题先检查 ABI 兼容性。

## 工作规范

### 决策点处理原则

实施过程中遇到任何需要做选择、但用户未明确指定的决策点时，**必须暂停当前工作，先向用户请示**，待获得明确指示后再继续执行。禁止自行假设或猜测用户的意图。常见决策点包括但不限于：算法参数取值、网络结构选择、实现方案取舍、版本兼容性处理方式等。

### 任务完成后回归测试

每个步骤的代码修改完成后，若当前项目已具备运行启动脚本的条件（即 Step 1 的环境搭建和依赖安装已完成），**必须从头执行启动脚本**（如 `train_reach_avoid.py`），验证本次修改未引入错误、脚本能正常走通。具体要求：

- 运行命令需包含完整的环境变量设置（`LD_LIBRARY_PATH`、`DISPLAY` 等）
- 若脚本报错，需定位并修复问题后重新运行，直到脚本完整执行通过
- 运行结果（成功或报错信息）需记录到 `plan.json` 的 `completed_work` 中
- 若因缺少外部资源（如 checkpoint 文件）导致无法完整运行，需在记录中注明，并说明已验证到哪一步

### 任务完成后必须记录

每次完成一个步骤或子任务后，**必须**在 `plan.json` 对应步骤的 `completed_work` 字段中记录：
- 完成了哪些具体工作
- 遇到的问题及解决方案
- 修改了哪些文件
- 是否通过验证（如 import 测试、训练运行等）

如果一个步骤分多次完成，每次都要追加记录，不要覆盖之前的内容。

### 任务完成汇报规范

每次完成一个步骤或子任务后，**必须**向用户输出以下内容（作为最终回复的一部分）：

1. **完成的工作**：具体做了哪些改动、修改了哪些文件、新增了哪些文件
2. **遇到的问题**：实施过程中碰到了什么困难或异常（如果没有则注明"无"）
3. **解决方案**：针对遇到的问题采取了什么措施、为什么选择该方案
4. **验证结果**：是否通过了该步骤的验证要求（如 import 测试、运行测试等）

同时，上述内容**必须**同步写入 `plan.json` 对应步骤的 `completed_work` 字段中。如果一个步骤分多次完成，每次都要追加记录（注明日期），不要覆盖之前的内容。

示例格式：
```
[2025-06-01] 第一次完成部分：
- 完成了 xxx，修改了 xxx 文件
- 遇到问题：xxx
- 解决方案：xxx
- 验证：import xxx 通过

[2025-06-02] 第二次完成部分：
- 完成了 xxx
- 无额外问题
- 验证：训练脚本可正常运行
```

### 依赖下载策略

安装 Python 依赖（pip install）时，**优先使用清华 PyPI 镜像源**加速下载：

```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果清华源中找不到对应版本、版本不匹配、或下载失败，则**自动切换回官方源**重试：

```bash
pip install <package> -i https://pypi.org/simple
```

同理，conda 安装时优先使用清华 conda 镜像，失败则回退到默认源。具体规则：
- 第一次尝试：清华源（`https://pypi.tuna.tsinghua.edu.cn/simple`）
- 若报错 `No matching distribution found` 或版本冲突：切换官方源（`https://pypi.org/simple`）重试
- 若官方源也失败：检查版本兼容性，必要时调整版本号（参考 AGENTS.md 中的版本约束）

### 代码修改原则

- **不破坏现有功能**：每次修改前确认当前代码能正常运行，修改后立即验证
- **最小化改动范围**：只修改当前步骤涉及的文件，不做无关重构
- **保留原有代码**：替换算法时，被替换的代码可以注释保留但不要删除，方便回退
- **对齐 JAX 参考实现**：算法移植时逐行对照 JAX 版本，确保数值行为一致



### 验证要求

- **Step 1**：`import isaacgym`、`import rsl_rl`、`import legged_gym` 无报错，`train_reach_avoid.py` 可正常运行
- **Step 2-3**：修改后运行训练脚本，确认环境能正常 reset/step，返回值维度正确
- **Step 4**：用随机数据对比 JAX 版和 PyTorch 版输出，容差 1e-5
- **Step 5**：实例化 `EC_EFPPO_ActorCritic`，调用 `act()`/`evaluate()` 确认输出维度和梯度流
- **Step 6**：单元测试 buffer 的 rollout 存储和 advantage 计算
- **Step 7**：端到端小规模训练（num_envs=64），确认三网络 loss 下降
- **Step 8**：全量训练，对比 EC-EFPPO 与基线的成功率和能量消耗

## 使用环境

```bash
# 激活环境
conda activate hdmcr
```

## 运行时注意事项

- **import 顺序**：必须先 `import isaacgym` 再 `import torch`，否则 CUDA 上下文冲突会导致进程挂起
- **环境变量**：运行前需设置 `LD_LIBRARY_PATH` 包含 conda env 的 lib 目录，以及 `DISPLAY=:0`
- **MKL 版本**：PyTorch 1.8.1 与新版 MKL 不兼容，需降级到 2021.4.0（已在 conda 环境中处理）

```bash
# 推荐的运行前设置
export LD_LIBRARY_PATH=/home/tanshan/miniconda3/envs/hdmcr/lib:$LD_LIBRARY_PATH
export DISPLAY=:0
conda activate hdmcr
```
