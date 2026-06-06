# HDMCRA Debug Log

本文档用于记录 HDMCRA 当前阶段的训练稳定性诊断。旧的实现计划和历史 debug 记录已经不再作为当前判断依据；后续分析从本文档重新开始。

## 当前状态

- EC-EFPPO 的主体实现已经完成，可以运行端到端 Go2 训练。
- 当前尚未证明训练可以稳定达到 Reach-Avoid PPO 基线水平。
- 后续工作重点是基于训练日志和代码链路，判断算法语义、环境信号、超参数、网络结构或统计口径中是否存在不合理设计。
- 任何改动都需要先形成可验证假设，再通过测试或训练结果验证。

## 当前待分析问题

| ID | 状态 | 严重性 | 问题 | 相关日志/文件 | 下一步 | 验证后下一步计划 |
|---|---|---|---|---|---|---|
| D001 | resolved | P0 | 首轮全量训练 success 低平台，critic loss 量级异常，entropy/std 不更新 | `legged_gym_go2/logs/ecfppo_go2/20260605-102937/training.log` | 已修复 `std` 未进入 policy optimizer，并补充训练诊断指标 | 已被 D002 取代：`std` 能更新后出现无界增长，需改为 `log_std` 参数化并限制探索噪声 |
| D002 | open | P0 | `std` 可更新后无界增长，entropy 推高探索噪声，success 早期峰值后坍塌 | `legged_gym_go2/logs/ecfppo_go2/20260605-231206/training.log` | 使用 `log_std -> exp` 参数化、限制 `std` 范围、降低并退火 entropy | 修改后跑 100-200 iter 诊断训练；若噪声受控但 reach/energy 仍异常，转向 critic target、energy 饱和和 done mask 语义 |

## 训练记录索引

| Run ID | 日期 | 命令/配置 | 迭代范围 | peak success | final success | 关键异常 | 结论 |
|---|---|---|---|---|---|---|---|
| 20260605-102937 | 2026-06-05 | `train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500` | 1-1500 | 0.194 | 0.171 | `reach_loss` 到 1e20 后仍停在 1e16；`energy_loss` 到 1e7；entropy 恒定 | 训练完整但未稳定收敛，存在明确实现/设计可疑点 |
| 20260605-231206 | 2026-06-05 | 修复 `std` optimizer 后再次训练 | 1-1387 | 0.339 | 约 0.096 | `std_mean` 从约 1.3 增至 100+；entropy 从约 4.3 增至 18+；`reach_loss` 延迟但仍到 1e19 | `std` optimizer 修复生效，但直接优化 std 会导致探索噪声失控，success 早期改善后坍塌 |

## 分析记录

### D001: 最新 EC-EFPPO 全量训练稳定性分析

- 日期：2026-06-05
- 状态：resolved
- 严重性：P0
- 触发原因：最新训练日志显示 success 长期低于预期，同时 value loss 量级严重异常。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260605-102937/training.log`
- 现象：
  - 日志完整，共 1500 轮，最终 checkpoint 正常保存。
  - `success` 从早期低值上升后进入平台期，最终为 0.171，峰值为 0.194（iter 1228）。
  - 分段均值：1-50 为 0.0428，201-300 为 0.0914，301-500 为 0.1645，之后基本停在 0.17-0.18。
  - `reach_loss` 在 iter 10 超过 1e6，iter 32 超过 1e9，iter 46 超过 1e12，iter 70 超过 1e15，iter 91 超过 1e16；最大值约 1.03e20（iter 293），最终仍约 1.70e16。
  - `energy_loss` 在 iter 248 超过 1e7，最终约 2.83e7，最大约 3.77e7（iter 630）。
  - `entropy` 全程恒定为 4.2568，`ent_coef` 全程为 0.01。
  - 最终 checkpoint 中 `std` 仍为 `[1.0, 1.0, 1.0]`。
  - `gamma_reach` 正常从 0.999 退火到 0.99999，并在约 iter 751 达到上限。
- 初步假设：
  - H1：`std` 没有加入 `policy_optimizer`，导致策略标准差不更新，entropy bonus 对策略探索基本不起作用。
  - H2：`targets_reach` 或 `calculate_reach_gae()` 输出量级异常，导致 reach critic loss 爆炸并污染 combined advantage 的策略信号。
  - H3：energy critic 的 target/输出尺度仍不一致，`energy_target_rms` 不能阻止 loss 进入 1e7 量级。
  - H4：success 平台期说明策略学到了一部分可达行为，但 critic/advantage 信号没有继续提供有效改进方向。
- 代码链路：
  - `train_ecfppo.py`：rollout 收集 `obs/g/h/energy/energy_consumption/dones`，记录 success 和 loss。
  - `EC_EFPPO_Buffer.compute_advantages()`：构造 `reach_append`、`energy_append`、`V_total_append`、`g_append`，计算三路优势。
  - `calculate_indexs3()`：计算 earliest-reach done 矩阵。
  - `calculate_reach_gae()`：生成 `targets_reach` 和 reach advantage。
  - `EC_EFPPO.update()`：计算 actor、energy critic、reach critic 三路 loss。
  - `EC_EFPPO.__init__()`：当前 `policy_optimizer` 只包含 `actor.parameters()`，不包含 `actor_critic.std`。
- 证据：
  - 日志统计显示 `entropy` 的唯一值为 4.2568。
  - 最终 checkpoint 的 `std` 为 `[1.0, 1.0, 1.0]`，说明训练 1500 轮后动作标准差完全没有变化。
  - 代码中 `policy_optimizer = Adam(self.actor_critic.actor.parameters())`，没有包含 `self.actor_critic.std`。
  - `reach_loss` 的爆炸发生很早，iter 91 已经达到 1e16，早于 success 平台稳定阶段。
  - `gamma_reach` 退火按预期完成，因此当前主要异常不是 gamma 没有更新。
- 结论：
  - 这次训练不是简单轮数不足，而是存在明确训练信号异常。
  - `std` 未被优化是一个确定的实现问题，会导致 entropy 项失效、探索尺度固定，应优先修复并补测试。
  - `reach_loss` 和 `energy_loss` 的量级异常需要进一步 dump target、value 和 advantage 的统计，不能只通过调 learning rate 判断。
- 改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：将 `actor_critic.std` 加入 `policy_optimizer` 和 policy 梯度裁剪范围，并在 optimizer step 后保持 `std > 0`。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：在 `compute_advantages()` 后保存 `debug_stats`，记录 value、target、advantage、done 和 energy 统计。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：按 `debug_stats_interval` 写入 `debug` 行，包含 `std`、`done_for_gae`、energy 饱和比例、reach value/target 和 advantage std。
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：为 EC-EFPPO 增加 `debug_stats_interval = 10`。
  - `tests/test_ecfppo.py`：补充 policy optimizer 必须包含并更新 `std` 的回归测试。
- 验证：
  - 已解析完整 `training.log`。
  - 已读取 `model_final.pt`，确认 `std` 未变化。
  - 已运行 `diagnose_advantage_detail.py` 诊断 `model_final.pt`，实际 horizon=200、64 envs。
  - 诊断中 `V_reach_append` 范围约为 `[-3.38e8, 1.08e9]`，而 `targets_reach` 主体仍在 g/h 语义尺度附近，说明 loss 爆炸的直接来源是 reach critic 输出发散后污染 bootstrap 和 combined advantage。
  - 诊断中 `advantages_total` 绝对均值约为 `1.41e8`，`Reach Advantage` 绝对均值约为 `1.43e8`，明显压过 energy advantage。
  - 诊断中 `done_for_gae mean=0.997`，每个环境 200 步中约 196-200 步被标记为 done，说明 earliest-index/done mask 对当前 Go2 能量状态非常激进。
  - 诊断中 `energy = -400` 比例约为 97.7%，`energy < 0` 比例约为 99.0%，说明当前固定 `std=1` 和动作裁剪后的能耗公式会让大多数 rollout 很快进入能量下界。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo.py`，结果 16 passed。
  - 已启动一次临时小规模训练 `--num_envs 64 --max_iterations 20`，为避免长时间占用 GPU 在 iter 13 后终止。
  - 临时训练日志 `logs/ecfppo_go2/20260605-230043/training.log` 显示 iter 10 时 `std_mean=1.2842`、`std_min=1.2683`、`std_max=1.2940`，说明 `std` 已随 policy optimizer 更新。
  - 同一 debug 行显示 `done_mean=0.9822`、`energy_min_ratio=0.9679`、`energy_neg_ratio=0.9855`，说明 energy 饱和和 done mask 过密问题在修复 `std` 后仍然存在。
- 后续动作：
  - 修复 `std` 未加入 `policy_optimizer` 的确定实现问题，并补测试确保 `std` 会被 optimizer 管理。
  - 在训练中增加受控诊断指标，统计 `targets_reach`、`values_reach`、`targets_energy`、`values_energy`、`advantages_total`、`done_for_gae`、`energy=-400` 比例和 `std`。
  - 使用小规模训练验证 `std` 是否更新、energy 是否仍快速饱和、reach critic 是否仍在早期发散。
- 验证后下一步计划：
  - 若 `std` 修复后 `entropy/std` 正常变化且 reach loss 明显下降，继续做 200-500 iter 中等规模训练确认 success 是否突破 0.18 平台。
  - 若 reach critic 仍快速进入 `1e6+`，优先给 reach critic 增加输出/target/bootstrapped value 的语义边界约束，或降低 reach critic 学习率。
  - 若 `energy=-400` 和 `done_for_gae` 仍接近全饱和，重新审查 energy 初始分布、`min_energy`、`energy_consumption_scale`、动作裁剪能耗和 earliest-index done 语义。


### D002: std 无界增长导致探索噪声失控

- 日期：2026-06-06
- 状态：open
- 严重性：P0
- 触发原因：修复 `std` optimizer 后，最新训练不再表现为 entropy 恒定，但出现 `std` 和 entropy 持续增大，success 早期提升后明显坍塌。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260605-231206/training.log`
- 现象：
  - 日志已跑到约 iter 1387。
  - `success` 在 iter 118 左右达到峰值约 0.339，之后逐步下降，后期约 0.07-0.10。
  - `std_mean` 从 iter 10 的约 1.31 增至 iter 150 的约 10.41，后期超过 100。
  - `entropy` 从约 4.3 持续增至约 18.27，说明 entropy bonus 正在持续鼓励更大的动作分布方差。
  - `reach_loss` 爆炸被延迟但没有消失：iter 77 超过 1e6，iter 405 超过 1e18，最大约 4.36e19。
  - `done_mean` 仍接近 1，`energy_min_ratio` 仍约 0.97，说明能量下界饱和和 done mask 过密仍是后续重点。
- 初步假设：
  - H1：直接优化实际 `std` 不够稳健；entropy 梯度会持续推大标准差，且没有上界。
  - H2：`entropy_coef=0.01` 对当前任务过强，持续探索奖励会压过已经学到的早期可达行为。
  - H3：`std` 失控会放大动作裁剪和能耗饱和，从而进一步恶化 `done_for_gae` 和 reach critic bootstrap。
  - H4：即使控制 `std`，reach critic 和 energy/done 语义仍可能存在独立问题，需要下一轮训练验证后再判断。
- 代码链路：
  - `EC_EFPPO_ActorCritic.update_distribution()`：根据 actor 均值和动作标准差构造 Normal 分布。
  - `EC_EFPPO.update()`：policy loss 中包含 `- entropy_coef * entropy`，会鼓励更高 entropy。
  - `train_ecfppo.py`：记录 `std_mean/std_min/std_max`、`entropy`、`done_mean`、`energy_min_ratio`、`v_reach_min`、`reach_loss`。
  - `GO2EC_EFPPOCfgPPO.algorithm`：控制初始动作噪声、entropy 系数和退火。
- 证据：
  - 修复 D001 后，`std` 不再固定为 1，证明 optimizer 修复有效。
  - 但最新日志中 `std_mean` 持续增至 100+，对应动作采样噪声远超动作裁剪范围，训练信号会被无意义探索主导。
  - 常见 PPO 连续动作实现通常优化 `log_std`，再通过 `exp(log_std)` 得到正标准差；这种参数化比直接优化实际 `std` 更常见，也更容易做范围约束。
- 结论：
  - D001 的最小修复暴露出第二个确定问题：EC-EFPPO 不能直接无界优化实际 `std`。
  - 当前优先级最高的修改不是继续调 critic，而是先让策略分布噪声受控，否则后续 reach/energy 诊断会被无界探索噪声污染。
- 改动：
  - `rsl_rl/rsl_rl/modules/actor_critic.py`：将 EC-EFPPO 动作噪声从 `std` 参数改为 `log_std` 参数，`std` 由 `exp(clamp(log_std))` 计算，保证标准差为正且有上下界。
  - `rsl_rl/rsl_rl/modules/actor_critic.py`：增加 `clamp_log_std_()`，并在 `load_state_dict()` 中兼容旧 checkpoint 的 `std` 字段。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：policy optimizer 改为管理 actor 参数和 `log_std`，policy step 后限制 `log_std` 范围。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：从配置读取 `init_noise_std`、`log_std_min`、`log_std_max`。
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：EC-EFPPO 设置 `init_noise_std=0.5`、`log_std_min=-2.0`、`log_std_max=0.0`，对应 `std` 约 `[0.135, 1.0]`；`entropy_coef` 从 0.01 降为 0.001，并启用 entropy 退火。
  - `tests/test_ecfppo_actor_critic.py`、`tests/test_ecfppo.py`、`tests/test_train_ecfppo.py`：更新并补充 `log_std`、旧 checkpoint 兼容、optimizer 管理和配置测试。
- 验证：
  - 已运行 `python3 -m py_compile` 检查修改过的 Python 文件。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo_actor_critic.py`，结果 13 passed。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo.py`，结果 16 passed。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_train_ecfppo.py'`，结果 12 passed。
  - `tests/test_energy_consumption_scale.py` 在设置 `LD_LIBRARY_PATH` 后进入 IsaacGym/PhysX 初始化，但发生 segmentation fault，暂记录为仿真扩展层运行失败，不作为本次 Python 逻辑回归失败。
- 后续动作：
  - 运行 100-200 iter 小规模诊断训练，优先观察 `std_mean/std_min/std_max` 是否稳定在 `[0.135, 1.0]` 内，`entropy` 是否不再单调无界增长。
  - 同时观察 `reach_loss` 是否仍在 100 iter 内超过 1e6，`v_reach_min` 是否快速进入极大负值。
  - 同时观察 `energy_min_ratio` 和 `done_mean` 是否仍接近 1。
- 验证后下一步计划：
  - 若 `std/entropy` 受控且 success 不再坍塌，继续跑 300-500 iter 中等规模训练确认是否突破 0.339 早期峰值并保持稳定。
  - 若 `std/entropy` 受控但 `reach_loss` 仍快速发散，下一步优先审查 reach critic 输出、target 和 bootstrap value 的边界约束。
  - 若 `std/entropy` 受控但 `energy_min_ratio` 和 `done_mean` 仍接近 1，下一步优先审查能耗尺度、`min_energy`、动作裁剪后的能耗计算和 earliest-index done 语义。
  - 若 `std` 被上界长期卡住且 success 无改善，需要重新评估 entropy 系数是否仍偏大，或策略均值学习是否被 critic advantage 噪声污染。

## 决策记录

- 2026-06-05：确认 `std` 未加入 policy optimizer 是确定实现 bug，已按最小修复处理。该改动不改变 EC-EFPPO 的 GAE/target 语义，只恢复参考实现中 policy 分布参数可训练的基本行为。
- 2026-06-05：新增诊断日志属于观测性改动，用于后续判断 reach critic 发散、energy 饱和和 `done_for_gae` 过密是否仍存在。
- 2026-06-05：临时训练已证明 `std` 修复生效，但没有证明训练稳定性已解决；下一轮应优先分析 energy 饱和和 done mask。
- 2026-06-06：确认直接优化实际 `std` 会造成探索噪声无界增长。EC-EFPPO 改为优化 `log_std`，通过 `exp(clamp(log_std))` 得到标准差，并降低/退火 entropy，先消除策略分布层面的不稳定来源。

## 记录模板

后续新增分析记录时使用下面的结构：

```markdown
### DXXX: 简短问题标题

- 日期：YYYY-MM-DD
- 状态：open / resolved / abandoned
- 严重性：P0 / P1 / P2
- 触发原因：哪次训练、哪条日志或哪个测试暴露了问题。
- 相关日志：`path/to/training.log`
- 现象：直接观察到的指标、曲线或错误。
- 初步假设：可能原因，必须可以验证。
- 代码链路：涉及的文件、函数和数据流。
- 证据：日志统计、张量统计、测试结果或代码对照。
- 结论：当前判断。
- 改动：如果做了修改，列出文件、旧值、新值和原因。
- 验证：运行的测试、训练命令和结果。
- 后续动作：下一步要做什么。
- 验证后下一步计划：做完本次修改并完成训练验证后，根据结果继续推进哪条决策路径。
```

<!--
示例记录，后续正式记录时可以参考这个结构，不要把本注释当作真实结论。

### D999: reach critic target 量级异常

- 日期：2026-06-06
- 状态：open
- 严重性：P0
- 触发原因：训练日志中 `reach_loss` 长期处于 1e16 量级，远高于可学习范围。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/<timestamp>/training.log`
- 现象：
  - `success` 在 0.16-0.19 附近波动。
  - `reach_loss` 维持在 1e16 量级。
  - `entropy` 基本不变化。
- 初步假设：
  - `calculate_reach_gae()` 生成的 `targets_reach` 量级异常。
  - `done_for_gae` 或 `h/g` 序列时间对齐存在问题。
  - reach critic value clipping 掩盖了 target 量级问题。
- 代码链路：
  - `train_ecfppo.py` rollout 收集 `g/h`。
  - `EC_EFPPO_Buffer.compute_advantages()` 构造 `reach_append` 和 `V_reach_append`。
  - `calculate_reach_gae()` 计算 `targets_reach`。
  - `EC_EFPPO.update()` 计算 `reach_loss`。
- 证据：
  - 待 dump `targets_reach.min/max/mean/std`。
  - 待 dump `values_h.min/max/mean/std`。
- 结论：待定。
- 改动：暂无。
- 验证：暂无。
- 后续动作：增加临时统计或单独脚本复现一个 rollout 的 advantage 量级。
- 验证后下一步计划：如果 target 确认异常，优先检查 GAE/done；如果 value 输出先发散，优先检查 critic 更新和输出约束。
-->
