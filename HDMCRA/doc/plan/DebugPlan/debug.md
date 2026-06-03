# HDMCRA 项目问题分析报告

> 基于 HDMCRA 代码、plan.json、两个参考仓库（JAX 版 Go2HierarchicalMiniCostReachAvoid 和 PyTorch 基线 Go2HierarchicalReachAvoidRL）的逐行对比分析。
>
> 分析日期：2026-06-03

---

## 一、致命 Bug（直接影响训练正确性）

### Bug 1：success rate 计算传入了错误的变量

**位置**：`legged_gym_go2/legged_gym/scripts/train_ecfppo.py:279`

**问题**：调用 `compute_reach_avoid_success_rate()` 时，`h_sequence` 参数位置传入的是 `alg.buffer.energy[1:]`，即能量消耗序列，而非安全约束序列。

**基线定义**（`Go2HierarchicalReachAvoidRL/legged_gym_go2/legged_gym/scripts/train_reach_avoid.py:51`）：
- `g < 0` → 到达目标
- `h >= 0` → 进入不安全区域
- 成功 = 先到达目标，且到达前从未违反约束

**影响**：所有 success_rate、exec_cost、avg_energy 统计值全部无效。plan.json Step 8 的成功率和能量结论应视为失效。

---

### Bug 2：buffer 根本没存 h_values

**位置**：`rsl_rl/rsl_rl/algorithms/ecfppo.py:79`

**问题**：EC_EFPPO_Buffer 只存了 `g_values`、`energy`、`energy_consumption`、`dones`，**没有 `h_values` 字段**。`add()` 接口不接受 h 参数，`train_ecfppo.py:243` 的 rollout 循环也没把 `h_vals` 写进 buffer。

**与 JAX 参考的差异**：JAX 版（`Go2HierarchicalMiniCostReachAvoid/rl/EC-EFPPO.py:183`）训练时并不按 g/h 算 success rate，而是记录 "not reaching goal" 和平均能耗。但 Go2 版新增的评估指标（success rate）依赖 g 和 h，当前实现无法支撑。

**修复范围**：
1. buffer 增加 `h_values` 存储
2. `add()` 接口增加 h 参数
3. rollout 循环写入 h
4. `compute_advantages()` 中 `calculate_indexs3` 的输入需要连通 h

---

### Bug 3：reset() 观测与能量不同步

**位置**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py:93-105`

**问题**：当前 reset 流程：
```python
obs = _compute_high_level_observations()   # 此时 energy 还是旧值/未初始化
self.energy = uniform(min_energy, max_energy)  # 采样新 energy
return obs  # 返回的是包含旧 energy 的观测
```

**JAX 参考实现**（`Go2HierarchicalMiniCostReachAvoid/env/reach_avoid/half_cheetah_avoid.py:36`，`pendulum_constraint.py:89`）：
```python
init_energy = jax.random.uniform(...)  # 先采样 energy
obs = ... + [init_energy]              # 再拼进 observation
return obs
```

**影响**：训练的第一个 timestep 拿到的 obs 里的 energy 值是上一个 episode 的残留值（或未初始化值），和实际 energy 状态不一致。

---

### Bug 4：能耗按未裁剪动作计算

**位置**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py:115,123`

**问题**：
- `update_velocity_commands()` 先把动作 clip 到 `[-1,1]` 再缩放执行
- `update_energy()` 用的是原始未裁剪的 `high_level_actions ** 2` 计算能耗

**JAX 参考**（`Go2HierarchicalMiniCostReachAvoid/env/reach_avoid/half_cheetah_avoid.py:50`）：先 tanh 约束再用变换后的动作算能耗。

**影响**：策略在"被截断执行"的同时承担更大的能耗惩罚，动作空间和能耗空间的映射关系不一致。

---

### Bug 5：action_repeat 下能耗累计缺失

**位置**：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py:140`

**问题**：plan.json（第 82 行）明确写了"每次低层执行都应贡献消耗"。但 `step()` 里 `update_energy()` 只在低层循环外调用一次。

**配置**：`high_level_action_repeat = 5`（`go2_config.py`）

**影响**：一个高层动作驱动 5 次低层步进，当前实现系统性低估总能耗为实际的 1/5。

---

## 二、设计偏差（与参考实现/计划不一致）

### 偏差 1：energy 归一化方案错误

**位置**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py:261`

**问题**：
- 当前做法：`energy / 400.0`
- 实际范围：min_energy=-400 → -1.0，max_energy=800 → 2.0，即 **[-1, 2]**
- plan.json 声称：归一化到 [-1, 1] ——**不满足**
- JAX 参考（`half_cheetah_avoid.py:45`，`pendulum_constraint.py:108`）：**直接拼原始 energy，不做归一化**

**结论**：这层归一化是 HDMCRA 自加的，既不是 JAX 对齐，范围也不对。

---

### 偏差 2：网络架构已从 JAX 对齐切到基线对齐

| 项目 | JAX 参考 | HDMCRA 计划（Step 5） | HDMCRA 实际 |
|------|----------|----------------------|-------------|
| 隐藏层 | 2x256 | 2x256 | 4x512 |
| 激活函数 | tanh | tanh | elu |
| 来源 | `actorcritic.py:93` | `plan.json:134` | `go2_config.py:232` |

实际配置对齐的是 Go2 基线（`train_reach_avoid.py:126`），不是 JAX。改动本身合理（Go2 任务更复杂），但 plan.json、README、测试还在声称"忠实 JAX 移植"。

---

### 偏差 3：超参数全面偏离 JAX 默认值

| 参数 | JAX 默认 | HDMCRA 实际 | 偏差 |
|------|----------|-------------|------|
| gamma_energy | 1.0 | 0.99 | 有折扣 vs 无折扣 |
| vf_coef | 0.5 | 1.0 | 翻倍 |
| learning_rate | 3e-4 | 1e-3 | 3.3x |
| network | 2x256+tanh | 4x512+elu | 完全不同 |

来源对比：
- JAX 默认：`Go2HierarchicalMiniCostReachAvoid/rl/arguments.py:18`
- HDMCRA 实际：`go2_config.py:238`

---

## 三、测试问题（验证链条失效）

### 问题 1：测试断言旧配置

**位置**：`tests/test_train_ecfppo.py:20`（`test_config_class`）

**问题**：测试断言 `gamma_energy == 1.0`、`vf_coef == 0.5`、`learning_rate == 3e-4`，但实际配置（`go2_config.py:238`）是 0.99、1.0、1e-3。

**影响**：plan.json 声称"9/9 回归测试全部通过"（`plan.json:204`），这只可能对应旧版本，不代表当前代码状态。

---

### 问题 2：test_energy_state.py 掩盖了 reset bug

**位置**：`tests/test_energy_state.py`（`test_energy_in_observation`）

**问题**：测试在检查 observation 最后一维是否等于 `energy / 400.0` 之前，手动再调用了一次 `_compute_high_level_observations()`。这绕过了 reset() 中"先算观测再采样 energy"的时序 bug，让测试通过但实际上掩盖了真实问题。

---

### 问题 3：没有测试覆盖 h_values 相关逻辑

整个测试套件（5 个文件，41 个测试）没有一个测试验证：
- h_values 是否被正确计算和传递
- success rate 是否正确使用 g 和 h
- buffer 是否存储了 h

---

### 问题 4：Step 8 "性能对比"从未完成

plan.json Step 8 标记为 completed，但注明"full training and cross-validation with JAX remain as follow-up work"。核心验证目标没完成就标记 completed，矛盾。

---

## 四、计划层面的问题

### 问题 1：Step 完成标准过于宽松

8 个 Step 全部标记 completed，但实际情况：

| Step | 计划目标 | 实际状态 |
|------|---------|---------|
| Step 5 | 2x256+tanh 对齐 JAX | 实际是 4x512+elu 对齐基线 |
| Step 7 | 超参数对齐 JAX | gamma_energy/vf_coef/LR 全部偏离 |
| Step 8 | full training + cross-validation | 未完成，仅 small-scale validation |

"completed" 的含义从"按计划实现"漂移成了"代码能跑"。

---

### 问题 2："JAX 对齐"叙述失控

从 Step 2 到 Step 8，plan.json 反复使用"与 JAX 对齐"的措辞。但中途决策（换网络、改超参、加归一化）已经实质性偏离了 JAX。文档没有同步更新，导致读者对项目状态产生错误认知。

---

### 问题 3：缺少"偏离记录"机制

当实现中途决定偏离原始计划时（比如换网络架构），没有在 plan.json 中记录：
- 为什么偏离
- 偏离了什么
- 新的验证标准是什么

---

## 五、已验证正确的部分

### 三网络架构拆分

三网络拆分（Policy + Energy Value + Reach Value）和三个独立优化器的设计是正确的，与 JAX 参考实现（`EFPPO_utils.py:35`）一致。`rsl_rl/modules/actor_critic.py:158` 的实现正确。

### GAE 三路计算

`ecfppo_gae.py` 中的 `calculate_indexs3`、`calculate_energy_gae`、`calculate_reach_gae` 移植自 JAX，测试覆盖了正常和边界情况。

### Buffer 和算法框架

`EC_EFPPO_Buffer` 和 `EC_EFPPO` 的整体结构（独立优化器、三路 PPO 更新、梯度裁剪）与 JAX 版对齐。

---

## 六、总体结论

当前项目的核心算法框架已经搭起来了，但**"框架能跑"和"实现正确"是两回事**。真正失真的地方集中在：

1. **环境-观测一致性**：reset 时序错误、能耗按未裁剪动作算、action_repeat 累计缺失
2. **评估指标正确性**：h_values 未存储、success rate 传参错误
3. **验证链条断裂**：测试断言旧值、掩盖真实 bug、h_values 无测试覆盖
4. **文档可信度**：plan.json completed 状态失真、"JAX 对齐"叙述过时

---

## 七、修复优先级

| 优先级 | 修复项 | 影响范围 | 涉及文件 |
|--------|--------|----------|----------|
| **P0** | buffer 存 h_values + success rate 用正确的 g/h | 所有评估指标失效 | `ecfppo.py`, `train_ecfppo.py` |
| **P0** | reset() 先采样 energy 再算观测 | 每个 episode 第一步训练信号错误 | `high_level_navigation_env.py` |
| **P1** | 能耗用裁剪后动作 + action_repeat 累计 | 能耗信号系统性偏差 | `high_level_navigation_env.py`, `hierarchical_go2_env.py` |
| **P1** | 测试对齐当前配置，补 h_values 测试 | 验证链条断裂 | `test_train_ecfppo.py`, `test_energy_state.py`, 新增测试 |
| **P2** | energy 归一化方案确认（保持/去掉/改范围） | observation 分布 | `high_level_navigation_env.py` |
| **P2** | plan.json 叙述更新，记录偏离决策 | 文档可信度 | `plan.json`, `README.md` |

---

## 八、详细修复方案

### Fix 1（P0）：success rate 传参错误 + buffer 缺 h_values

这两个问题需要一起修，因为 success rate 依赖 h_values，而 buffer 没存。

#### 1.1 buffer 增加 h_values 存储

**文件**：`rsl_rl/rsl_rl/algorithms/ecfppo.py`

在 `EC_EFPPO_Buffer.__init__()` 中（第 82 行 `self.g_values` 之后）增加：

```python
self.h_values = torch.zeros(horizon + 1, num_envs, device=device)
```

#### 1.2 add() 接口增加 h 参数

**文件**：`rsl_rl/rsl_rl/algorithms/ecfppo.py`，`EC_EFPPO_Buffer.add()`

签名增加 `h_values`、`next_h` 两个参数：

```python
def add(
    self,
    obs, actions, log_probs, values, value_reach,
    energy, energy_consumption, g_values, h_values,  # 新增 h_values
    dones,
    next_obs, next_energy, next_g, next_h,           # 新增 next_h
) -> None:
```

方法体增加存储：

```python
self.h_values[idx] = h_values
# ...
self.h_values[idx + 1] = next_h
```

#### 1.3 rollout 循环写入 h

**文件**：`legged_gym_go2/legged_gym/scripts/train_ecfppo.py`

在训练循环的 rollout 部分（第 244 行 `alg.buffer.add(...)`），将已有的 `h_vals` 和 `next_h` 传入：

```python
alg.buffer.add(
    obs=obs,
    actions=actions,
    log_probs=log_probs,
    values=values_energy,
    value_reach=values_reach,
    energy=energy,
    energy_consumption=energy_consumption,
    g_values=g_vals,
    h_values=h_vals,          # 新增
    dones=dones,
    next_obs=next_obs,
    next_energy=next_energy,
    next_g=next_g,
    next_h=next_h,            # 新增
)
```

注意：当前代码第 234 行已经解包了 `next_h`，只是没传进 buffer。

#### 1.4 修正 success rate 调用

**文件**：`legged_gym_go2/legged_gym/scripts/train_ecfppo.py`

将第 279-283 行：

```python
success_rate, execution_cost, avg_energy = compute_reach_avoid_success_rate(
    alg.buffer.g_values[1:],       # [T, N] 跳过初始状态
    alg.buffer.energy[1:],         # ← 错误：这是 energy 不是 h
    energy_sequence=alg.buffer.energy,
)
```

改为：

```python
success_rate, execution_cost, avg_energy = compute_reach_avoid_success_rate(
    alg.buffer.g_values[1:],       # [T, N] g 序列
    alg.buffer.h_values[1:],       # [T, N] h 序列（修正）
    energy_sequence=alg.buffer.energy,  # [T+1, N] 完整能量序列
)
```

#### 1.5 修正 mini_training_loop 测试

**文件**：`tests/test_train_ecfppo.py`，`test_mini_training_loop()`

在 buffer.add() 调用中增加 `h_values` 和 `next_h`：

```python
h_vals = torch.ones(num_envs) * -1.0  # 初始安全
next_h = torch.ones(num_envs) * -1.0

alg.buffer.add(
    # ... 已有参数 ...
    h_values=h_vals,      # 新增
    next_h=next_h,        # 新增
)
```

---

### Fix 2（P0）：reset() 观测与能量不同步

**文件**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`

将 `reset()` 方法（第 80-105 行）的执行顺序调整为：**先采样 energy，再算观测**。

当前代码：
```python
def reset(self):
    base_obs = self.base_env.reset()
    self._compute_high_level_observations()          # ← 此时 energy 是旧值
    self.energy.uniform_(self.min_energy, self.max_energy)
    self.energy_consumption.zero_()
    initial_avoid_metric, initial_reach_metric = self._get_current_metrics()
    initial_g_values = self._compute_g_function(initial_reach_metric)
    initial_h_values = self._compute_h_function(initial_avoid_metric)
    return self.high_level_obs_buf, initial_g_values, initial_h_values, self.energy.clone()
```

修正为：
```python
def reset(self):
    base_obs = self.base_env.reset()

    # 1. 先采样 energy（与 JAX 参考一致）
    self.energy.uniform_(self.min_energy, self.max_energy)
    self.energy_consumption.zero_()

    # 2. 再算观测（此时 energy 是新值，会正确拼入 obs 末尾）
    self._compute_high_level_observations()

    # 3. 计算初始 g/h
    initial_avoid_metric, initial_reach_metric = self._get_current_metrics()
    initial_g_values = self._compute_g_function(initial_reach_metric)
    initial_h_values = self._compute_h_function(initial_avoid_metric)

    return self.high_level_obs_buf, initial_g_values, initial_h_values, self.energy.clone()
```

#### 修正 test_energy_state.py

**文件**：`tests/test_energy_state.py`，`test_energy_in_observation()`

当前代码手动再调用了一次 `_compute_high_level_observations()` 来绕过时序 bug。修正后 reset() 已经保证同步，测试应直接检查 reset 返回的 obs：

```python
def test_energy_in_observation():
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    obs, _, _, energy = nav_env.reset()  # reset 返回的 obs 应该包含正确的 energy
    assert torch.allclose(obs[:, -1], energy / 400.0, atol=1e-5)
```

注意：如果后续决定去掉 energy 归一化（Fix 6），这里也要相应修改。

---

### Fix 3（P1）：能耗按未裁剪动作计算

**文件**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`

将 `update_energy()` 方法（第 123-138 行）改为先 clip 再计算：

当前代码：
```python
def update_energy(self, high_level_actions):
    consumption = torch.sum(high_level_actions ** 2, dim=1) * self.energy_consumption_scale
    self.energy_consumption = consumption.clone()
    self.energy = torch.clamp(self.energy - consumption, self.min_energy, self.max_energy)
```

修正为：
```python
def update_energy(self, high_level_actions):
    # 先 clip 到 [-1, 1]，与 update_velocity_commands() 的实际执行动作一致
    clipped_actions = torch.clip(high_level_actions, -1.0, 1.0)
    consumption = torch.sum(clipped_actions ** 2, dim=1) * self.energy_consumption_scale
    self.energy_consumption = consumption.clone()
    self.energy = torch.clamp(self.energy - consumption, self.min_energy, self.max_energy)
```

这样能耗惩罚和实际执行的命令对齐，策略不会为被截断的动作承担额外惩罚。

#### 修正相关测试

**文件**：`tests/test_energy_state.py`，`test_energy_consumption_formula()`

当前测试传入 `actions = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]`，这些值都在 [-1,1] 范围内，所以 clip 不影响结果。但如果要测试 clip 行为，应增加一个用例：

```python
def test_energy_clip_action():
    """超出 [-1,1] 的动作应被 clip 后再算能耗"""
    nav_env = HighLevelNavigationEnv(MockBaseEnv(), HighLevelNavigationConfig())
    nav_env.reset()
    actions = torch.tensor([[2., 0., 0.]])  # 超出范围
    nav_env.update_energy(actions)
    # clip 后等效于 [1,0,0]，消耗 = 1^2 * 8 = 8.0，而不是 2^2 * 8 = 32.0
    assert abs(nav_env.energy_consumption[0].item() - 8.0) < 1e-5
```

---

### Fix 4（P1）：action_repeat 下能耗累计缺失

**文件**：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

将 `step()` 方法（第 125-185 行）中的能耗计算从循环外移到循环内。

当前代码（第 140-141 行）：
```python
# 1. Compute energy consumption for this high-level action
self.high_level_env.update_energy(high_level_actions)

# 2. Update desired velocity commands
self.high_level_env.update_velocity_commands(high_level_actions)
# ...
for _ in range(self.low_level_action_repeat):
    # 低层执行（多次）
    ...
```

修正为：
```python
# 1. Update desired velocity commands（先确定执行什么）
self.high_level_env.update_velocity_commands(high_level_actions)
desired_velocity_commands = self.base_env.commands[:, :3].clone()

# 2. Execute low-level policy multiple times, accumulating energy each time
base_obs = None
base_infos = None
avoid_metric = None
reach_metric = None
aggregated_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
total_energy_consumption = torch.zeros(self.num_envs, device=self.device)

for _ in range(self.low_level_action_repeat):
    self.base_env.commands[:, :3] = desired_velocity_commands
    self.base_env.compute_observations()
    current_base_obs = self.base_env.get_observations()
    with torch.no_grad():
        low_level_actions = self.low_level_policy(current_base_obs)
    base_obs, privileged_obs, _, step_dones, base_infos, avoid_metric, reach_metric = self.base_env.step(
        low_level_actions
    )
    aggregated_dones |= step_dones.bool()

# 3. 累计能耗：每次低层执行都消耗能量（对齐计划文档）
self.high_level_env.update_energy(high_level_actions)
# 注意：如果计划要求每次低层执行都独立计算能耗，
# 则需要在循环内调用 update_energy，并累加 energy_consumption。
# 当前简化方案：高层动作执行一次，能耗按一次计算，
# 但 energy_consumption_scale 应乘以 low_level_action_repeat。
```

更精确的方案是让 `update_energy()` 接受一个 `repeat` 参数：

```python
def update_energy(self, high_level_actions, repeat=1):
    clipped_actions = torch.clip(high_level_actions, -1.0, 1.0)
    consumption = torch.sum(clipped_actions ** 2, dim=1) * self.energy_consumption_scale * repeat
    self.energy_consumption = consumption.clone()
    self.energy = torch.clamp(self.energy - consumption, self.min_energy, self.max_energy)
```

然后在 `hierarchical_go2_env.py` 的 `step()` 中：

```python
# 循环结束后，按 repeat 次数累计能耗
self.high_level_env.update_energy(high_level_actions, repeat=self.low_level_action_repeat)
```

---

### Fix 5（P1）：测试对齐当前配置 + 补 h_values 测试

#### 5.1 修正 test_config_class 断言

**文件**：`tests/test_train_ecfppo.py`，`test_config_class()`

将第 22-31 行的断言改为匹配当前 `GO2EC_EFPPOCfgPPO` 的实际值：

```python
def test_config_class():
    cfg = GO2EC_EFPPOCfgPPO()
    assert cfg.algorithm.gamma_energy == 0.99        # 修正：原断言 1.0
    assert cfg.algorithm.gamma_reach_init == 0.999
    assert cfg.algorithm.gamma_reach_final == 0.99999
    assert cfg.algorithm.gae_lambda == 0.95
    assert cfg.algorithm.clip_eps == 0.2
    assert cfg.algorithm.vf_coef == 1.0              # 修正：原断言 0.5
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.anneal_entropy == False
    assert cfg.algorithm.max_grad_norm == 0.5
    assert cfg.algorithm.learning_rate == 1e-3       # 修正：原断言 3e-4
    assert cfg.algorithm.num_learning_epochs == 10
    assert cfg.algorithm.num_mini_batches == 8
    assert cfg.runner.experiment_name == 'ecfppo_go2'
    print("[PASS] test_config_class")
```

#### 5.2 修正 test_alg_with_config 断言

**文件**：`tests/test_train_ecfppo.py`，`test_alg_with_config()`

```python
assert alg.gamma_energy == 0.99     # 修正：原断言 1.0
assert alg.gamma_reach_init == 0.999
assert alg.clip_param == 0.2
assert alg.value_loss_coef == 1.0   # 修正：原断言 0.5
assert alg.anneal_entropy == False
```

#### 5.3 新增 h_values 相关测试

**文件**：`tests/test_train_ecfppo.py`，新增：

```python
def test_buffer_stores_h_values():
    """验证 buffer 正确存储 h_values"""
    from rsl_rl.algorithms.ecfppo import EC_EFPPO_Buffer
    num_envs, horizon, obs_dim, act_dim = 4, 8, 10, 3
    buf = EC_EFPPO_Buffer(num_envs, horizon, (obs_dim,), (act_dim,), torch.device('cpu'))

    for step in range(horizon):
        buf.add(
            obs=torch.randn(num_envs, obs_dim),
            actions=torch.randn(num_envs, act_dim),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            value_reach=torch.randn(num_envs),
            energy=torch.rand(num_envs) * 400,
            energy_consumption=torch.rand(num_envs) * 5,
            g_values=torch.randn(num_envs),
            h_values=torch.randn(num_envs),      # 新增
            dones=torch.zeros(num_envs),
            next_obs=torch.randn(num_envs, obs_dim),
            next_energy=torch.rand(num_envs) * 400,
            next_g=torch.randn(num_envs),
            next_h=torch.randn(num_envs),         # 新增
        )

    assert buf.h_values.shape == (horizon + 1, num_envs)
    assert buf.step == horizon
    print("[PASS] test_buffer_stores_h_values")


def test_success_rate_uses_h_values():
    """验证 success rate 正确使用 h_values（h >= 0 表示不安全）"""
    from legged_gym.scripts.train_ecfppo import compute_reach_avoid_success_rate

    T, N = 10, 4
    # 全部到达目标
    g_seq = torch.ones(T, N) * -0.5
    # 前2个环境在 t=3 进入不安全区域
    h_seq = torch.ones(T, N) * -1.0
    h_seq[3:, :2] = 0.5  # h >= 0 → 不安全

    success_rate, _, _ = compute_reach_avoid_success_rate(g_seq, h_seq)
    # 只有后2个环境成功（到达目标前未违反约束）
    assert success_rate == 0.5, f"expected 0.5, got {success_rate}"
    print("[PASS] test_success_rate_uses_h_values")
```

---

### Fix 6（P2）：energy 归一化方案确认

当前做法 `energy / 400.0` 存在两个问题：
1. 范围是 [-1, 2]，不是声称的 [-1, 1]
2. JAX 参考不做归一化

**推荐方案**：去掉归一化，直接用原始 energy 值拼入 observation。

**文件**：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`

将第 261-264 行：

```python
# Append normalized energy to the end of observations
energy_normalized = self.energy / 400.0
self.high_level_obs_buf[:, -1] = energy_normalized
```

改为：

```python
# Append raw energy to the end of observations（与 JAX 参考一致，不做归一化）
self.high_level_obs_buf[:, -1] = self.energy
```

**影响**：observation 的数值范围会变化（从 [-1,2] 变为 [-400,800]），可能需要调整网络初始化或学习率。如果训练不稳定，可考虑用 `energy / max_energy` 归一化到 [-0.5, 1.0]。

**替代方案**：如果保留归一化，至少修正范围。用 `max_energy` 做分母：

```python
energy_normalized = self.energy / self.max_energy  # 范围: [-0.5, 1.0]
```

无论哪种方案，都要同步更新 `tests/test_energy_state.py` 中的 `test_energy_in_observation`。

---

### Fix 7（P2）：plan.json 叙述更新

**文件**：`doc/plan/AchievePlan/plan.json`

需要更新的内容：

1. **Step 5**：标注实际网络架构为 4x512+elu（对齐 Go2 基线），而非计划的 2x256+tanh（对齐 JAX）。记录偏离原因："Go2 任务比 JAX 参考的简单环境更复杂，需要更大网络容量"。

2. **Step 7**：标注超参数已偏离 JAX 默认值，记录实际值（gamma_energy=0.99, vf_coef=1.0, LR=1e-3）和偏离原因："对齐 Go2 基线以解决 EC-EFPPO 成功率过低问题"。

3. **Step 8**：将 "completed" 改为 "in_progress" 或 "partially_completed"。标注：
   - small-scale validation 已完成
   - full training 未完成
   - cross-validation with JAX 未完成
   - success rate 统计存在 bug（Bug 1+2），已有数据不可信

4. **全局**：将 "与 JAX 对齐" 的叙述改为 "基于 JAX 参考实现，适配 Go2 任务的 PyTorch 移植版"。

---

## 九、修复执行顺序建议

```
Phase 1（P0，先修这两个才能拿到正确的训练数据）:
  ├─ Fix 1: buffer 存 h_values + success rate 修正
  │   ├── ecfppo.py: buffer 增加 h_values，add() 增加 h 参数
  │   ├── train_ecfppo.py: rollout 写 h，success rate 用 h
  │   └── test_train_ecfppo.py: 修正 mini_training_loop 测试
  │
  └─ Fix 2: reset() 时序修正
      ├── high_level_navigation_env.py: 先采样 energy 再算 obs
      └── test_energy_state.py: 去掉手动再调用 _compute_high_level_observations

Phase 2（P1，修正能耗信号）:
  ├─ Fix 3: 能耗用裁剪后动作
  │   ├── high_level_navigation_env.py: update_energy() 先 clip
  │   └── test_energy_state.py: 增加 clip 测试用例
  │
  └─ Fix 4: action_repeat 累计
      ├── high_level_navigation_env.py: update_energy() 增加 repeat 参数
      └── hierarchical_go2_env.py: step() 传入 repeat

Phase 3（P1，测试对齐）:
  └─ Fix 5: 测试断言修正 + 新增 h_values 测试
      └── test_train_ecfppo.py: 修正 3 个断言，新增 2 个测试

Phase 4（P2，可选）:
  ├─ Fix 6: energy 归一化方案确认
  └─ Fix 7: plan.json 叙述更新
```
