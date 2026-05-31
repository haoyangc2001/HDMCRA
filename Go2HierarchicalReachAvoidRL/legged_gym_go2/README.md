# legged_gym_go2/ 目录说明

本目录是项目的核心工程包，基于 [legged_gym](https://github.com/littlebearqqq/legged_gym_go2) 和 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 改造而来，提供 Unitree Go2 四足机器人的完整训练与部署栈，包括 IsaacGym 仿真环境、分层 RL 架构、Reach-Avoid 导航任务，以及 MuJoCo 仿真和真实硬件的部署脚本。

## 目录结构

```
legged_gym_go2/
├── setup.py                        # pip 安装脚本（包名 unitree_rl_gym）
├── legged_gym/                     # 主 Python 包
│   ├── envs/                       # 环境实现
│   │   ├── base/                   # 基础类（所有机器人共享）
│   │   ├── go2/                    # Unitree Go2 环境 + 分层导航
│   │   ├── g1/                     # Unitree G1 人形机器人
│   │   ├── h1/                     # Unitree H1 人形机器人
│   │   ├── h1_2/                   # Unitree H1-2 人形机器人
│   │   └── tinymal/                # TinyMal 四足机器人
│   ├── scripts/                    # 训练/测试/可视化脚本
│   └── utils/                      # 工具函数
├── deploy/                         # 部署相关
│   ├── deploy_mujoco/              # MuJoCo 仿真部署
│   ├── deploy_real/                # 真实硬件部署
│   └── pre_train/                  # 预训练低层策略 checkpoint
├── resources/                      # 机器人 URDF/XML 模型 + 网格
│   └── robots/                     # go2, g1, h1, h1_2, tinymal
└── tmp_urdf_assets/                # 临时障碍物 URDF（圆柱体等）
```

## 环境层级架构

```
┌──────────────────────────────────────────────────────────────────┐
│  scripts/train_reach_avoid.py  (训练入口)                         │
│  └── ReachAvoidPPO  ←→  HierarchicalVecEnv                       │
├──────────────────────────────────────────────────────────────────┤
│  go2/hierarchical_go2_env.py::HierarchicalGO2Env                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 高层: go2/high_level_navigation_env.py::HighLevelNavigationEnv│ │
│  │   观测: [heading, velocity, target_dir, lidar]               │ │
│  │   动作: [vx, vy, vyaw] 速度指令                              │ │
│  │   g/h 值: 到达/安全指标                                      │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ 低层: 预训练运动策略 (deploy/pre_train/go2/policy_1.pt)       │ │
│  │   观测: [关节状态, IMU, 速度指令]                             │ │
│  │   动作: 12维关节目标位置                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  go2/go2_env.py::GO2Robot → base/legged_robot.py::LeggedRobot    │
│  → base/base_task.py::BaseTask → IsaacGym 仿真                   │
└──────────────────────────────────────────────────────────────────┘
```

## 环境实现详解

### `envs/base/` — 基础类

| 文件 | 类 | 功能 |
|---|---|---|
| `base_config.py` | `BaseConfig` | 所有配置类的基类，支持 `to_dict()` 序列化 |
| `base_task.py` | `BaseTask` | IsaacGym 仿真初始化、场景创建、渲染控制 |
| `legged_robot.py` | `LeggedRobot(BaseTask)` | 四足机器人通用逻辑：物理仿真步进、力矩计算、奖励函数管理、观测计算、地形高度采样 |
| `legged_robot_config.py` | `LeggedRobotCfg` / `LeggedRobotCfgPPO` | 环境和训练的完整配置（200+ 参数），涵盖：环境、初始状态、物理、观测噪声、奖励、归一化、PPO 超参等 |
| `origin_legged_robot.py` | — | 原始 legged_robot 的备份 |

`LeggedRobot.step()` 的核心流程：
1. 裁剪动作 → `decimation` 次物理仿真（控制频率 < 物理频率）
2. 计算 `post_physics_step()`：刷新状态、计算奖励、检查终止
3. 计算观测 `compute_observations()`
4. 返回 `(obs, privileged_obs, reward, done, info)`

### `envs/go2/` — Go2 分层导航（核心）

| 文件 | 类 | 功能 |
|---|---|---|
| `go2_config.py` | `GO2RoughCfg` | Go2 环境配置：4096 并行环境、45维观测、12维动作、12个关节默认角度、9个障碍物位置、目标区域等 |
| | `GO2RoughCfgPPO` | Go2 训练配置：4层×1024 MLP、5000 迭代、horizon=24、mini-batch=24576 |
| | `GO2HighLevelCfg` | 高层导航配置：继承 `GO2RoughCfg`，覆盖障碍物/目标参数 |
| | `GO2HighLevelCfgPPO` | 高层训练配置：4层×512 MLP、5000 迭代 |
| `go2_env.py` | `GO2Robot(LeggedRobot)` | Go2 环境扩展：重写 `step()` 返回额外的 `avoid_metric` 和 `reach_metric`，重写 `reset()` |
| `hierarchical_go2_env.py` | `HierarchicalGO2Env` | 分层封装：加载预训练低层策略、高层动作重复执行、g/h 值转换 |
| `high_level_navigation_env.py` | `HighLevelNavigationEnv` | 高层导航环境：观测计算、速度指令转换、g/h 函数、lidar 编码 |

### `envs/g1/` `envs/h1/` `envs/h1_2/` `envs/tinymal/` — 其他机器人

均继承 `LeggedRobot`，各自实现：
- 噪声向量 `_get_noise_scale_vec()`
- 观测计算 `compute_observations()`
- 独立的配置文件 `*_config.py`

## scripts/ — 训练与可视化脚本

| 脚本 | 功能 |
|---|---|
| `train_reach_avoid.py` | Reach-Avoid PPO 训练主入口，创建分层环境并驱动训练循环 |
| `play_reach_avoid.py` | 加载训练好的 checkpoint，可视化分层导航策略的执行效果 |
| `test_reach_avoid.py` | 加载 checkpoint，收集轨迹数据并保存为 JSON 文件（用于后续分析） |
| `plot_env_layout.py` | 绘制环境俯视布局图（障碍物、目标、轨迹），支持叠加多条轨迹 |
| `plot_training_results.py` | 解析训练日志，绘制成功率、执行成本、策略损失等指标曲线 |
| `play_fixed_commands_with_video.py` | 用固定速度指令测试低层策略并录制视频 |

## deploy/ — 部署

### `deploy/deploy_mujoco/` — MuJoCo 仿真部署

使用 MuJoCo 加载机器人 XML 模型，运行预训练的低层策略进行仿真验证：

- `deploy_mujoco.py` — 基础部署框架
- `deploy_mujoco_go2.py` — Go2 专用部署
- `deploy_mujoco_tinymal.py` — TinyMal 专用部署
- `configs/*.yaml` — 各机器人的部署配置（关节 PD 参数、默认角度、观测维度等）

### `deploy/deploy_real/` — 真实硬件部署

通过 `unitree_sdk2_python` 与真实机器人通信：

- `deploy_real.py` — 真实硬件部署主脚本
- `config.py` — 部署配置数据类
- `common/command_helper.py` — 电机指令构造（阻尼模式、零力矩、位置控制）
- `common/remote_controller.py` — 遥控器按键映射
- `common/rotation_helper.py` — IMU 重力方向计算、坐标变换
- `configs/*.yaml` — G1/H1/H1-2 的真实部署配置

### `deploy/pre_train/` — 预训练低层策略

| checkpoint | 机器人 |
|---|---|
| `go2/policy_1.pt` | Go2 低层运动策略（高层导航调用此 checkpoint） |
| `g1/motion.pt` | G1 运动策略 |
| `h1/motion.pt` | H1 运动策略 |
| `h1_2/motion.pt` | H1-2 运动策略 |

## resources/ — 机器人模型

| 目录 | 机器人 | 格式 | 关节数 |
|---|---|---|---|
| `robots/go2/` | Unitree Go2 | URDF + MuJoCo XML | 12 |
| `robots/g1_description/` | Unitree G1 | URDF + MuJoCo XML | 12/23/29 DOF 多种配置 |
| `robots/h1/` | Unitree H1 | URDF + MuJoCo XML | — |
| `robots/h1_2/` | Unitree H1-2 | URDF | 12/标准 DOF |
| `robots/tinymal/` | TinyMal | URDF + MuJoCo XML | — |

## utils/ — 工具函数

| 文件 | 功能 |
|---|---|
| `task_registry.py` | 任务注册表：`register()` 注册环境类和配置，`make_env()` 创建环境实例 |
| `helpers.py` | 参数解析、配置覆盖、seed 设置、checkpoint 路径查找 |
| `math.py` | 数学工具：`wrap_to_pi` 等 |
| `isaacgym_utils.py` | IsaacGym 张量工具：`get_euler_xyz` 从四元数提取欧拉角 |
| `logger.py` | 日志工具 |
| `terrain.py` | 地形管理 |

## tmp_urdf_assets/ — 临时障碍物

用于导航任务的圆柱体障碍物 URDF 模型，不同颜色和尺寸：

- `cylinder_r0.3_h0.6_rgba*.urdf` — 小圆柱（半径 0.3m，高 0.6m）
- `cylinder_r30_h60_rgba*.urdf` — 大圆柱（半径 30m，高 60m，用作边界墙）

## 安装

```bash
# 安装 legged_gym_go2 包
cd legged_gym_go2
pip install -e .

# 依赖
# - isaacgym（需从 NVIDIA 获取，见 ../isaacgym/）
# - rsl_rl（见 ../rsl_rl/）
# - mujoco==3.2.3
# - numpy==1.20
# - matplotlib, pyyaml, tensorboard
```

## 使用示例

```bash
# 训练高层 Reach-Avoid 导航策略
python legged_gym/scripts/train_reach_avoid.py

# 可视化训练好的策略
python legged_gym/scripts/play_reach_avoid.py

# 收集轨迹数据
python legged_gym/scripts/test_reach_avoid.py --checkpoint logs/<experiment>/model_final.pt

# 绘制环境布局和轨迹
python legged_gym/scripts/plot_env_layout.py --trajectories trajectories.json

# MuJoCo 仿真部署（低层策略）
python deploy/deploy_mujoco/deploy_mujoco_go2.py go2.yaml

# 真实硬件部署
python deploy/deploy_real/deploy_real.py
```

## 与项目其他部分的关系

```
Go2HierarchicalReachAvoidRL/
├── legged_gym_go2/    ← 本目录：环境、训练、部署
├── rsl_rl/            ← Reach-Avoid PPO 算法（被 train_reach_avoid.py 调用）
├── isaacgym/          ← NVIDIA GPU 物理仿真（被 envs/base/ 使用）
└── unitree_rl_gym/    ← unitree_rl_gym 原始模板（本目录的上游）
```

本目录是 `unitree_rl_gym` 的改造版本，主要改动：
- 新增 `go2/hierarchical_go2_env.py` 和 `high_level_navigation_env.py`（分层导航）
- 新增 `scripts/train_reach_avoid.py`（Reach-Avoid 训练循环）
- 在 `GO2Robot` 中扩展 `avoid_metric` 和 `reach_metric` 的计算
- 新增 `GO2HighLevelCfg` / `GO2HighLevelCfgPPO` 配置
