# 训练脚本说明

本目录包含 EC-EFPPO 算法在不同环境下的训练启动脚本。所有脚本均调用 `./src/rl/EC-EFPPO.py` 作为入口，通过命令行参数指定环境、超参数和训练配置。

## 脚本列表

| 脚本 | 环境 | 任务描述 |
|---|---|---|
| `run_pendulum_constraint.sh` | PendulumConstraint | 摆锤在受限力矩下摆起，最小化能量消耗 |
| `run_hopper_reach.sh` | HopperReach | Hopper 机器人到达目标位置 |
| `run_hopper_avoid_ceiling.sh` | HopperAvoidCeiling | Hopper 跳跃时避免头部碰触天花板 |
| `run_half_cheetah_avoid.sh` | HalfCheetahAvoid | HalfCheetah 奔跑时避免肢体进入禁区 |
| `run_wind_field.sh` | WindField | 四旋翼在风场中导航避障，最小化能量 |
| `run_f16_avoid.sh` | F16Avoid | F16 战机在约束下避障飞行 |

## 通用超参数说明

所有脚本共享以下核心参数，仅数值因环境复杂度不同而有差异：

### PPO 核心参数

| 参数 | 含义 | 所有脚本取值 |
|---|---|---|
| `--LR` | 学习率 | 3e-4 |
| `--UPDATE_EPOCHS` | 每次 PPO 更新的 epoch 数 | 10 |
| `--CLIP_EPS` | PPO 裁剪阈值 ε | 0.2 |
| `--GAE_LAMBDA` | GAE λ 参数 | 0.95 |
| `--MAX_GRAD_NORM` | 梯度裁剪范数 | 0.5 |
| `--ACTIVATION` | 激活函数 | tanh |
| `--ANNEAL_LR` | 启用学习率线性退火 | 所有脚本均启用 |
| `--ANNEAL_ENT` | 启用熵系数线性退火 | 所有脚本均启用 |

### 能量约束 / Reach-Avoid 参数

| 参数 | 含义 | 所有脚本取值 |
|---|---|---|
| `--GAMMA_ENERGY` | 能量 value function 折扣因子 | 1.0（无折扣，精确累加） |
| `--GAMMA_REACH_INIT` | reach 函数折扣因子初始值 | 0.99 |
| `--GAMMA_REACH_FINAL` | reach 函数折扣因子终止值 | 0.99 |

### 训练规模参数

| 参数 | 含义 |
|---|---|
| `--NUM_ENVS` | 并行环境数量 |
| `--NUM_STEPS` | 每条轨迹的时间步数 |
| `--TOTAL_TIMESTEPS` | 总训练时间步数 |
| `--STEP_SCAN` | 每个外层 step 内执行的内层训练循环次数 |
| `--NUM_MINIBATCHES` | minibatch 数量 |
| `--VF_COEF` | value function loss 权重 |
| `--ENT_COEF` | 熵正则化系数（退火起始值） |
| `--DIR` | checkpoint 和可视化输出的子目录名 |
| `--NAME` | wandb 日志中的实验名称 |
| `--CUDA_USE` | 可见 GPU 编号 |

## 各脚本配置差异

### `run_pendulum_constraint.sh`

- **并行环境少、轨迹长**：`NUM_ENVS=32`，`NUM_STEPS=400`
- **训练量最大**：`TOTAL_TIMESTEPS=80_000_000`（其他环境均为 20M）
- **minibatch 少**：`NUM_MINIBATCHES=8`
- **value loss 权重较低**：`VF_COEF=0.5`
- **多 GPU**：`CUDA_USE=0,1,2,3`

摆锤是最简单的环境（1维动作、4维观测），但需要更长的训练来收敛精确的能量约束策略。

### `run_hopper_reach.sh` / `run_hopper_avoid_ceiling.sh`

- **并行环境适中**：`NUM_ENVS=128`
- **轨迹较长**：`NUM_STEPS=400`
- **多 GPU**：`CUDA_USE=0,1,2,3`
- `run_hopper_reach.sh` 的 `STEP_SCAN=1`（每次只做 1 个内层循环），`run_hopper_avoid_ceiling.sh` 的 `STEP_SCAN=4`

Hopper 基于 brax 物理引擎，14维观测，3维动作。

### `run_half_cheetah_avoid.sh`

- **并行环境较多**：`NUM_ENVS=512`
- **轨迹较短**：`NUM_STEPS=200`
- **minibatch 较多**：`NUM_MINIBATCHES=32`
- **单 GPU**：`CUDA_USE=0`

HalfCheetah 是高维动作空间（6维），20维观测。

### `run_wind_field.sh`

- **并行环境最多**：`NUM_ENVS=1536`
- **STEP_SCAN 最大**：`STEP_SCAN=40`（内层循环最密集）
- **minibatch 最多**：`NUM_MINIBATCHES=64`
- **单 GPU**：`CUDA_USE=2`
- **独有参数**：`--SECTION=0`（选择风场区域）

WindField 是最复杂的环境（12维物理状态 + 四旋翼动力学 + 风场扰动），需要大量并行环境和密集训练。

### `run_f16_avoid.sh`

- **并行环境最多之一**：`NUM_ENVS=2048`
- **轨迹较短**：`NUM_STEPS=200`
- **STEP_SCAN 适中**：`STEP_SCAN=10`
- **单 GPU**：`CUDA_USE=0`

F16 是高维观测（26维），训练规模较大。

## 使用方式

从项目根目录运行：

```bash
# 示例：训练摆锤约束环境
bash script/run_pendulum_constraint.sh

# 示例：训练风场环境
bash script/run_wind_field.sh
```

训练产物（checkpoint、可视化图片）保存到 `model/{DIR}/` 下。训练日志同步上报到 wandb，项目名为 `EC-EFPPO-{EXP_NAME}`。

## 自定义训练

如需修改训练配置，有两种方式：

1. **修改脚本**：直接编辑对应 `.sh` 文件中的参数值
2. **命令行覆盖**：在脚本基础上追加参数，例如：
   ```bash
   bash script/run_pendulum_constraint.sh --LR=1e-4 --SEED=42
   ```
