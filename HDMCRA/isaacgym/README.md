# isaacgym/ 目录说明

本目录是 NVIDIA Isaac Gym 的本地安装副本（版本 `1.0.preview4`），为 Go2 分层导航系统提供 GPU 加速的物理仿真能力。原生 README 仅一行：`Please refer to docs/index.html to get started`。

## 目录结构

```
isaacgym/
├── python/                  # Python 包（核心）
│   ├── setup.py             # pip 安装脚本
│   ├── rlgpu_conda_env.yml  # conda 环境定义
│   └── isaacgym/            # Python 模块
│       ├── gymapi.py        # 主 API 绑定（加载 C++ 原生库）
│       ├── gymtorch.py      # PyTorch 张量互操作
│       ├── gymutil.py       # 可视化工具（坐标轴、线框等）
│       ├── torch_utils.py   # 四元数运算等数学工具
│       ├── terrain_utils.py # 地形生成工具
│       ├── rlgpu.py         # RL GPU 专用绑定
│       └── _bindings/       # C++ 原生库（.so/.pyd）及 USD 依赖
├── assets/                  # 物理模型资产
│   ├── mjcf/                # MuJoCo 格式模型
│   ├── urdf/                # URDF 格式模型
│   └── textures/            # 纹理贴图
├── docs/                    # HTML 文档（可本地浏览）
├── docker/                  # Docker 构建脚本
├── licenses/                # 第三方许可证
└── create_conda_env_rlgpu.sh  # conda 环境创建脚本
```

## Python 模块说明

### `gymapi.py` — 主 API 入口

根据当前 Python 版本（3.7/3.8）动态加载对应的 C++ 原生库（`gym_37.so` 或 `gym_38.so`），将所有底层符号注入全局命名空间。这是 Isaac Gym 的核心入口，提供：

- 物理仿真器创建与配置
- Actor（刚体/关节体）的加载与管理
- 传感器数据获取
- 渲染与可视化

### `gymtorch.py` — PyTorch 互操作

运行时编译 C++ 扩展（`gymtorch.cpp`），实现 Isaac Gym 内部数据与 PyTorch CUDA 张量的零拷贝共享。核心功能：

- `torch_jit` 将仿真状态直接映射为 PyTorch 张量
- 支持 GPU pipeline（数据留在 GPU 上，不经过 CPU）
- 自动检测 PyTorch 版本并适配 ABI

### `gymutil.py` — 可视化工具

提供调试用的几何图形绘制：

- `AxesGeometry` — 坐标轴（RGB = XYZ）
- `WireframeBoxGeometry` — 线框立方体
- 其他线段/点几何体

### `torch_utils.py` — 数学工具

GPU 加速的四元数/向量运算，使用 `@torch.jit.script` 编译：

- `quat_mul` — 四元数乘法
- `quat_apply` — 四元数旋转向量
- `quat_rotate` — 四元数旋转
- `normalize` — 向量归一化
- `to_torch` — numpy → torch 张量转换

### `terrain_utils.py` — 地形生成

程序化生成各种训练地形：

- `random_uniform_terrain` — 均匀噪声地形
- `sloped_terrain` — 坡度地形
- 其他（阶梯、崎岖等）

这些工具用于 legged_gym 的地形课程训练。

### `rlgpu.py` — RL GPU 绑定

与 `gymapi.py` 类似的动态加载机制，但加载的是 RL 专用的原生库（`rlgpu_37.so` / `rlgpu_38.so`），提供额外的 RL 训练支持功能。

## assets/ — 物理模型资产

### URDF 模型（`assets/urdf/`）

| 模型 | 描述 | 本项目用途 |
|---|---|---|
| `ball.urdf` | 球体 | 障碍物/目标标记 |
| `cube.urdf` | 立方体 | 通用物体 |
| `cartpole.urdf` | 倒立摆 | 基础控制任务 |
| `anymal_b_simple_description/` | ANYmal 四足机器人 | 参考（未直接使用） |
| `franka_description/` | Franka Panda 机械臂 | 操作任务 |
| `kinova_description/` | Kinova 机械臂 | 操作任务 |
| `kuka_allegro_description/` | Kuka + Allegro 灵巧手 | 操作任务 |
| `ycb/` | YCB 物体数据集（肉罐、香蕉、杯子、砖块） | 抓取任务 |
| `sektion_cabinet_model/` | IKEA 柜子 | 操作任务 |
| `nut_bolt/` | 螺母螺栓 | 精细操作任务 |
| `tray/` | 托盘 | 操作任务 |
| `objects/` | 简单物体（球、方块） | 通用 |

### MJCF 模型（`assets/mjcf/`）

| 模型 | 描述 |
|---|---|
| `nv_humanoid.xml` | NVIDIA 人形机器人 |
| `nv_ant.xml` | NVIDIA 蚂蚁机器人 |
| `humanoid_CMU_V2020_v2.xml` | CMU 人形机器人 |
| `open_ai_assets/fetch/` | Fetch 机器人（reach、push、pick_and_place 等任务） |
| `open_ai_assets/hand/` | Shadow Hand 灵巧手（操作任务） |

### 纹理（`assets/textures/`）

8 张环境纹理贴图（金属锈蚀、石板、木纹、墙面等），用于仿真场景渲染。

## 环境安装

### conda 环境

```bash
# 方式一：使用脚本
bash create_conda_env_rlgpu.sh

# 方式二：手动
conda env create -f python/rlgpu_conda_env.yml
conda activate rlgpu
cd python && pip install -e .
```

环境要求：
- Python 3.7（conda YAML 中指定）
- PyTorch 1.8.1 + CUDA 11.1
- 依赖：scipy、pyyaml、tensorboard

### Docker

```bash
cd docker
bash build.sh
bash run.sh
```

## 与项目其他部分的关系

```
legged_gym_go2/legged_gym/envs/go2/go2_env.py
    └── import isaacgym
        ├── isaacgym.gymapi   → 创建仿真、加载 URDF/MJCF 模型
        ├── isaacgym.gymtorch → 状态数据与 PyTorch 张量共享
        └── isaacgym.gymutil  → 调试可视化

legged_gym_go2/legged_gym/envs/base/legged_robot.py
    └── 使用 terrain_utils 生成训练地形

assets/urdf/ball.urdf 等
    └── 被 go2_env.py 作为障碍物/目标加载到仿真场景中
```

## 注意事项

- Isaac Gym 是 NVIDIA 的预览版产品，需要 NVIDIA GPU 支持
- 原生库（`_bindings/` 目录下的 `.so` 文件）为预编译二进制，不可修改
- docs/ 目录包含完整的 HTML 文档，可在浏览器中打开 `docs/index.html` 查看 API 参考
- 本项目实际使用的资产主要是 `ball.urdf`（障碍物）和基础几何体，Go2 机器人模型由 legged_gym 自行加载
