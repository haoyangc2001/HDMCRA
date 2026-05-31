# Reach-Avoid PPO Algorithm Design

## 摘要
本项目将 Go2 高层导航建模为一个带安全约束的 Reach-Avoid 决策问题。与标准 PPO 直接最大化累计奖励不同，这里的训练信号由目标函数 `g(s)` 与安全函数 `h(s)` 共同定义，并通过 Reach-Avoid 价值目标驱动策略网络与价值网络同步更新。对应实现主要位于 `rsl_rl/rsl_rl/algorithms/reach_avoid_ppo.py`、`rsl_rl/rsl_rl/modules/actor_critic.py` 和 `legged_gym_go2/legged_gym/scripts/train_reach_avoid.py`。

## 1. 问题定义

### 1.1 Reach-Avoid 目标
给定状态序列 \(s_0, s_1, \dots, s_T\) 与动作序列 \(a_0, a_1, \dots, a_{T-1}\)，智能体需要同时满足两类要求：

- Reach：最终进入目标区域。
- Avoid：在进入目标前始终保持安全，不碰撞障碍物。

项目中不直接依赖标量 reward，而是使用两个辅助函数：

\[
g(s_t) =
\begin{cases}
g_{\text{target}} < 0, & s_t \in \mathcal{G} \\
\alpha_g \cdot \text{dist}(s_t, \mathcal{G}), & s_t \notin \mathcal{G}
\end{cases}
\]

\[
h(s_t) =
\begin{cases}
h_{\text{safe}} < 0, & s_t \in \mathcal{S} \\
h_{\text{unsafe}} \ge 0, & s_t \notin \mathcal{S}
\end{cases}
\]

这里：

- \(g(s_t)\) 越小表示离目标越近，进入目标区后直接为负。
- \(h(s_t)\) 是安全硬约束，若 \(h(s_t)\ge0\) 则判定为安全违规。

对应环境实现位于 `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`。默认参数包括：

- `g_target_value = -300.0`
- `g_distance_scale = 100.0`
- `h_safe_value = -300.0`
- `h_unsafe_value = 300.0`

### 1.2 终止条件
训练中的终止标志不是单纯沿用环境 `done`，而是把安全约束违规纳入回合结束条件：

\[
d_t = d_t^{env} \lor [h(s_t) \ge 0]
\]

因此只要碰撞或进入不安全区域，轨迹在算法层面就会被视为终止。训练脚本还会把 rollout 最后一个时间步强制置为终止，以保证价值回传边界清晰。

## 2. Reach-Avoid 价值目标

### 2.1 单步 Reach-Avoid 目标
算法的核心是用“安全优先”的方式构造时间步价值候选项：

\[
Q_t^{(k)} = \max\left(h_t,\ \min\left(g_t,\ \gamma V_{t+1}^{(k)}\right)\right)
\]

该式有三层含义：

1. `min(g_t, γ V_{t+1}^{(k)})` 在“当前已经足够接近目标”与“继续依赖未来价值”之间取更优的 Reach 结果。
2. `max(h_t, ·)` 强制安全约束覆盖 Reach 收益；一旦 \(h_t \ge 0\)，该步候选值会被安全违规主导。
3. 候选值不只有一个，因为实现里会维护一个 `value_table`，包含从一步回看、多步回看得到的未来价值候选。

这对应 `_calculate_reach_gae()` 中的实现：

```python
vhs_row = torch.maximum(
    h_seq[idx].unsqueeze(0),
    torch.minimum(g_seq[idx].unsqueeze(0), gamma_tensor * value_table),
)
```

### 2.2 最终 Critic 目标
对每个时间步，代码会构造一组不同回溯深度的 \(Q_t^{(k)}\)，再用 GAE 风格的归一化系数做加权平均：

\[
\hat{Q}_t = \sum_{k=0}^{K} w_{t,k} Q_t^{(k)}, \qquad \sum_{k=0}^{K} w_{t,k}=1
\]

最终优势定义为：

\[
A_t = \hat{Q}_t - V_\phi(s_t)
\]

其中：

- \(\hat{Q}_t\) 是 Reach-Avoid critic 的训练目标。
- \(V_\phi(s_t)\) 是当前价值网络对状态的估计。
- \(A_t\) 决定策略网络更新方向。

## 3. Reach-Avoid GAE 原理

### 3.1 系数递推机制
标准 GAE 通过 \(\lambda\) 让多步回报在偏差与方差之间折中。本项目保留这一思想，但递推的对象不再是 reward-to-go，而是 Reach-Avoid 价值候选。

实现中维护 `gae_coeffs`，从后往前递推。关键步骤如下：

1. 将旧系数沿时间维滚动一位，为当前时间步腾出位置。
2. 如果前一时刻未终止，则使用 \(\lambda\) 做指数衰减。
3. 如果前一时刻已终止，则使用 \(\lambda/(1-\lambda)\) 重新初始化，保证新段轨迹的系数分布合理。
4. 如果当前时刻终止，则对应未来项整体清零。
5. 令 `gae_coeffs[0] = 1.0`，确保当前步始终参与目标构造。
6. 用 `coeff_sum` 做归一化，获得 \(\{w_{t,k}\}\)。

### 3.2 为什么这样设计
这种设计解决了两个问题：

- 安全终止会打断普通累计回报传播，而 Reach-Avoid 需要在终止边界上保留“安全优先”的硬约束。
- 高层导航的目标达成往往跨多个时间步，多步平滑能降低方差，避免仅凭瞬时 `g/h` 波动更新策略。

## 4. 策略网络原理

### 4.1 网络结构
策略网络定义在 `rsl_rl/rsl_rl/modules/actor_critic.py` 的 `ActorCritic.actor` 中。它是一个前馈 MLP：

- 输入：高层观测 `observations`
- 隐层：可配置全连接层，训练脚本里默认设为 `[512, 512, 512, 512]`
- 激活：由配置给定，当前训练脚本通常使用 `ReLU`
- 输出：动作均值 `mean`

高层动作维度是导航速度命令，例如 \([v_x, v_y, \omega_z]\)。网络输出并不是离散动作概率，而是连续动作分布的均值。

### 4.2 高斯策略分布
策略采用对角高斯分布：

\[
\pi_\theta(a_t|s_t) = \mathcal{N}(\mu_\theta(s_t), \sigma^2)
\]

对应实现：

```python
mean = self.actor(observations)
self.distribution = Normal(mean, mean * 0. + self.std)
```

其中：

- `mean = μ_θ(s_t)` 由 actor MLP 输出。
- `self.std` 是可学习参数，和状态无关，但随训练更新。
- `act()` 通过 `self.distribution.sample()` 采样动作。
- `get_actions_log_prob()` 计算旧策略和新策略下动作的对数概率，用于 PPO 比率。

### 4.3 策略更新目标
策略更新仍使用 PPO 的 clipped surrogate 目标，只是优势项来自 Reach-Avoid critic：

\[
L^{clip}(\theta) =
\mathbb{E}_t \left[
\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)
\right]
\]

\[
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
\]

在实现中，优势会先做标准化，以减少 batch 间尺度差异。随后以“最小化损失”的方式写成策略损失，因此代码内部会对优势符号做转换。

### 4.4 策略网络在 Reach-Avoid 任务中的作用
策略网络学习的不是“尽量多拿奖励”，而是：

- 选择让 `g(s)` 更快减小的高层速度命令。
- 同时避免产生会让 `h(s)` 变成非负的动作。
- 在有风险的状态下，通过 critic 给出的高风险优势信号主动回避障碍物。

## 5. 价值网络原理

### 5.1 网络结构
价值网络定义在 `rsl_rl/rsl_rl/modules/actor_critic.py` 的 `ActorCritic.critic` 中。它与 actor 分离，采用独立 MLP：

- 输入：critic 观测 `critic_observations`
- 隐层：训练脚本默认 `[512, 512, 512, 512]`
- 输出：单标量 \(V_\phi(s_t)\)

对应实现：

```python
value = self.critic(critic_observations)
```

由于 actor 和 critic 参数解耦：

- actor 专注于连续动作分布建模；
- critic 专注于拟合 Reach-Avoid 目标；
- 两者可共享输入维度，但不共享最后的表示头。

### 5.2 Critic 学习目标
critic 不回归普通累计回报，而是回归前面定义的 Reach-Avoid 目标 \(\hat{Q}_t\)。也就是说，critic 估计的是“在当前状态下，综合安全约束与目标到达难度后的风险敏感价值”。

这使得 \(V_\phi(s_t)\) 具有以下性质：

- 若当前状态接近碰撞边界，则估计值会被 \(h_t\) 拉高。
- 若状态安全且接近目标，则估计值会更多反映 \(g_t\) 的下降趋势。
- 若未来路径仍有明显不确定性，则估计值会依赖折扣后的未来值传播。

### 5.3 Clipped Value Loss
价值网络训练采用 PPO 风格的 clipped value loss：

\[
L^V(\phi)=\frac{1}{2}\mathbb{E}_t\left[
\max\left((V_\phi(s_t)-\hat{Q}_t)^2,\ (V_\phi^{clip}(s_t)-\hat{Q}_t)^2\right)
\right]
\]

其中裁剪版本抑制 critic 单次更新幅度过大，降低值函数震荡对优势估计的反噬。

## 6. 熵正则与探索
为了防止连续动作策略过早塌缩为近乎确定性策略，总损失中还加入熵正则项：

\[
L(\theta,\phi) = -L^{clip}(\theta) + c_v L^V(\phi) - c_e \mathbb{E}_t[H(\pi_\theta)]
\]

其中：

- `c_v` 是价值损失权重；
- `c_e` 是熵系数；
- \(H(\pi_\theta)\) 是高斯策略熵。

当 `entropy_coef` 较大时，策略会保持更强的探索；当其趋近 0 时，训练更偏向利用当前最优行为。

## 7. 训练数据流与缓冲区设计

### 7.1 Rollout Buffer
`ReachAvoidBuffer` 会存储：

- `observations`
- `actions`
- `log_probs`
- `values`
- `g_values`
- `h_values`
- `dones`

其中 `g_values` 与 `h_values` 是 Reach-Avoid PPO 相比标准 PPO 多出的关键字段。buffer 先保存完整 rollout，再统一计算 `advantages` 和 `returns`。

### 7.2 Batch 组织
`ReachAvoidBatch` 把训练所需数据封装成扁平批次，便于：

- 多轮 epoch 更新；
- mini-batch 随机采样；
- 对策略与价值网络重复利用同一批 rollout 数据。

## 8. 与高层导航环境的耦合关系

### 8.1 观测
高层策略看到的不是原始低层电机状态，而是导航相关观测，包括：

- 朝向编码 `cos(heading)` 与 `sin(heading)`
- 机体速度
- 与目标的相对方向和归一化距离
- 障碍物手工 lidar 编码

这些特征在 `legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` 中构造，目的在于让高层策略专注于几何导航决策。

### 8.2 动作
高层动作是速度命令，由分层环境送给底层 locomotion policy。底层策略负责把速度命令转换为稳定步态，所以 Reach-Avoid PPO 只优化“往哪里走、以多快速度走”，不直接优化 12 维关节控制。

## 9. 整体训练流程
在 `legged_gym_go2/legged_gym/scripts/train_reach_avoid.py` 中，训练流程可以概括为：

1. 创建 `HierarchicalGO2Env` 并封装为 `HierarchicalVecEnv`。
2. 初始化 `ActorCritic` 与 `ReachAvoidPPO`。
3. 在每个 rollout 内：
   - actor 根据当前观测采样动作；
   - 环境返回 `next_obs`、`next_g`、`next_h` 和 `dones`；
   - buffer 存储轨迹。
4. rollout 结束后，用最后时刻的 critic 值估计补全边界。
5. 调用 `compute_advantages()` 生成 Reach-Avoid `advantages` 与 `returns`。
6. 按 PPO 方式进行多 epoch、多 mini-batch 更新。
7. 记录成功率、执行成本、价值损失、策略损失等日志。

## 10. 与标准 PPO 的核心差异

| 维度 | 标准 PPO | 本项目 Reach-Avoid PPO |
|---|---|---|
| 训练信号 | reward | `g(s)` 与 `h(s)` |
| 终止条件 | 环境 `done` | 环境 `done` 或 `h(s) >= 0` |
| critic 目标 | 折扣回报 | Reach-Avoid 加权目标 \(\hat{Q}_t\) |
| advantage | reward-based GAE | Reach-Avoid GAE |
| 安全建模 | 通常靠奖励惩罚 | 直接作为硬约束并进入值目标 |

## 11. 代码对应关系

- Reach-Avoid 目标与 GAE：`rsl_rl/rsl_rl/algorithms/reach_avoid_ppo.py`
- 策略网络与价值网络：`rsl_rl/rsl_rl/modules/actor_critic.py`
- 训练循环与日志：`legged_gym_go2/legged_gym/scripts/train_reach_avoid.py`
- `g/h` 构造与导航观测：`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py`
- 分层动作执行：`legged_gym_go2/legged_gym/envs/go2/hierarchical_go2_env.py`

## 12. 实践含义
这个算法设计的核心收益在于：高层导航策略不再通过“调奖励函数”间接表达安全目标，而是直接把到达目标与避障安全写进值函数目标中。这样做通常会带来三点好处：

- 安全违规在优化中具有更明确的优先级；
- critic 学到的是更贴合任务定义的价值，而非混合奖励的经验近似；
- 在复杂障碍环境中，策略更新方向更稳定，也更容易解释。
