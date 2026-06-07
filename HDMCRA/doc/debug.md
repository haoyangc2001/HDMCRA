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
| D002 | open | P0 | `std` 可更新后无界增长，entropy 推高探索噪声，success 早期峰值后坍塌 | `legged_gym_go2/logs/ecfppo_go2/20260605-231206/training.log` | 已用 `log_std -> exp` 和上界控制探索噪声；最新训练证明 std 已受控但 success 仍坍塌 | 转入 D003：定位 reach target/bootstrap 发散与 energy/done 饱和的因果关系 |
| D003 | open | P0 | reach target/bootstrap 发散与 energy/done 饱和 | `legged_gym_go2/logs/ecfppo_go2/20260606-103051/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260606-230145/training.log` | 已确认极端 target 来自极少量 non-done/open bootstrap，已增加 reach bootstrap value 语义裁剪 | 跑 100-200 iter 大批量诊断训练，观察 `reach_clip_ratio`、`t_open_min`、`reach_loss` 和 success 是否改善；若仍失败，转向 energy/done 语义 |

## 训练记录索引

| Run ID | 日期 | 命令/配置 | 迭代范围 | peak success | final success | 关键异常 | 结论 |
|---|---|---|---|---|---|---|---|
| 20260605-102937 | 2026-06-05 | `train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500` | 1-1500 | 0.194 | 0.171 | `reach_loss` 到 1e20 后仍停在 1e16；`energy_loss` 到 1e7；entropy 恒定 | 训练完整但未稳定收敛，存在明确实现/设计可疑点 |
| 20260605-231206 | 2026-06-05 | 修复 `std` optimizer 后再次训练 | 1-1387 | 0.339 | 约 0.096 | `std_mean` 从约 1.3 增至 100+；entropy 从约 4.3 增至 18+；`reach_loss` 延迟但仍到 1e19 | `std` optimizer 修复生效，但直接优化 std 会导致探索噪声失控，success 早期改善后坍塌 |
| 20260606-103051 | 2026-06-06 | `log_std` 上界、低 entropy 并退火后的训练 | 1-446 | 0.307 | 0.042 | `std` 在 iter 30 后稳定为 1.0；`done_mean≈0.993-0.998`；`energy_min_ratio≈0.972`；`reach_loss` 到 4.39e18 | 探索噪声已受控，但 reach target/bootstrap 发散和 energy/done 饱和仍主导训练失败 |
| 20260606-143037 | 2026-06-06 | `--headless --num_envs 16 --max_iterations 50`，D003 分组 debug | 1-50 | 0.375 | 0.125 | 小规模未复现 reach loss 爆炸；`done_mean=0.9625-0.9981`；`energy_min_ratio=0.9602-0.9701`；`std_mean` 最终 0.9962 | 分组 debug 可用；energy/done 饱和稳定存在，早期最小 target 可来自极少量 non-done 样本 |
| 20260606-230145 | 2026-06-06 | 大批量训练，包含 `open_ratio/t_done/t_open/tmin_src` 分组字段 | 1-1286 | 0.307 | 0.013 | `tmin_src done` 全为 0；`tmin_t` 全为 199；`t_done_min=-300` 稳定；`t_open_min` 最低约 -8.59e9；`open_ratio` 仅约 0.0023-0.0072 | 极端 reach target 不是 done 分支或原始 `g/h` 造成，而是极少量 non-done/open 样本从发散的 reach value bootstrap 进入 target |

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


### D003: reach target/bootstrap 发散与 energy/done 饱和

- 日期：2026-06-06
- 状态：open
- 严重性：P0
- 触发原因：`log_std` 上界控制后，最新训练仍出现 success 坍塌、reach loss 爆炸和 energy/done 饱和。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260606-103051/training.log`
- 现象：
  - 当前日志解析到 iter 446，共 446 条 iter 记录和 44 条 debug 记录。
  - `std_mean` 从 iter 10 的 0.6034 上升到 iter 30 的 1.0000，之后稳定在上界 1.0000，说明 D002 的 `log_std` 限幅生效。
  - `entropy` 在 iter 23 后稳定为 4.2568，对应 `std=1.0` 的上界熵；这不是旧版 std 未训练，而是被上界控制后的结果。
  - `success` 峰值为 0.307（iter 29），后续坍塌，iter 446 为 0.042。
  - `reach_loss` 在 iter 24 超过 1e6，iter 43 超过 1e12，iter 335 超过 1e18，最大约 4.39e18。
  - `done_mean` 全程约 0.993-0.998，`energy_min_ratio` 全程约 0.972，说明大部分 rollout 时间步被 done mask 截断，同时能量长期卡在下界。
  - `v_reach_min` 和 `t_reach_min` 同步扩散到 1e9 量级，例如 iter 440 时分别约为 -8.264e9 和 -7.732e9。
- 初步假设：
  - H1：极端 `targets_reach` 主要来自少量 `done_for_gae == 0` 的 bootstrap 路径，critic 输出发散后污染下一轮 target。
  - H2：若 `done_for_gae == 1` 组也出现极端 target，则 `calculate_reach_gae()` 的 done 语义或时间对齐存在实现问题。
  - H3：`energy_min_ratio≈0.972` 说明 energy 下界饱和是独立异常，会让 `calculate_indexs3()` 和 combined advantage 的语义持续退化。
  - H4：当前不应继续调 entropy/std；策略分布层面的无界探索已经被控制，剩余问题在 critic target、done mask 和 energy 语义。
- 代码链路：
  - `EC_EFPPO_Buffer.compute_advantages()`：构造 `done_for_gae`，计算 reach、energy 和 combined advantage。
  - `calculate_indexs3()`：根据 energy、energy consumption 和 reach 序列生成 earliest-index done 矩阵。
  - `calculate_reach_gae()`：使用 `done_for_gae`、`h/g` 序列和 `V_reach_append` 生成 reach target。
  - `HierarchicalGO2Env.step()` 和 `HighLevelNavigationEnv.update_energy()`：每个高层动作按裁剪后动作平方消耗能量，并将 energy clamp 到下界。
- 证据：
  - D002 修改后 `std` 不再无界增长，排除了上一轮主要干扰项。
  - `done_mean` 和 `energy_min_ratio` 从 iter 10 开始就接近 1，早于 reach loss 进入 1e12 量级。
  - `t_reach_min` 随 `v_reach_min` 一起进入巨大负值，说明 target 已被 bootstrap value 污染，而不是单纯环境 `g/h` 原始量级异常。
- 结论：
  - 当前最优先不是继续长训或调学习率，而是增加分组 debug，定位极端 target 来自 done 组、non-done 组还是 min target 的特定 bootstrap 来源。
  - 本轮代码只增加诊断统计，不改变训练语义，保证下一次短训能回答具体因果问题。
- 改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：新增 masked debug stats，分别统计 `done_for_gae == 1` 与 `done_for_gae == 0` 下的 `targets_reach`、`values_reach` 和 `advantages_total`。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：记录 `targets_reach_min` 对应的时间步、done 标记、当前/下一步 reach value、`g/h` 和 energy。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：在 debug 行输出 `open_ratio`、`t_done`、`t_open`、`v_open`、`tmin_src` 和 `adv_open_std`。
- 验证：
  - 已运行 `python3 -m py_compile rsl_rl/rsl_rl/algorithms/ecfppo.py legged_gym_go2/legged_gym/scripts/train_ecfppo.py`，语法检查通过。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo.py`，结果 16 passed。
  - 已尝试运行 `train_ecfppo.py --num_envs 64 --max_iterations 60` 和 `--headless --num_envs 64 --max_iterations 60`；两次日志实际写入项目根目录 `logs/ecfppo_go2/`，但因误查旧路径被提前终止，只得到 iter 14/19 的不完整日志。
  - 已完成 50 iter 诊断训练：`logs/ecfppo_go2/20260606-143037/training.log`，命令为 `--headless --num_envs 16 --max_iterations 50`。
  - 该小规模训练未复现大规模 reach loss 爆炸：`reach_loss` 范围约 4.13e4 到 1.63e5，final 7.53e4。
  - `std_mean` 从 0.6003 增至 0.9962，仍受上界控制；`entropy` final 4.2433。
  - `done_mean` 始终很高，范围约 0.9625-0.9981；`energy_min_ratio` 始终约 0.9602-0.9701，说明 energy/done 饱和在小规模训练中仍稳定存在。
  - iter 10 和 iter 20 的 `targets_reach_min` 来自 `done_for_gae == 0` 的 open 样本；iter 30/40/50 的最小 target 来自 done 样本，但数值仍在约 `[-300, 1138]` 的语义范围内。
  - 结论：新增分组 debug 能正常工作；小规模短训不适合验证 reach loss 爆炸，但足以确认 energy/done 饱和和 open 样本来源统计。
- 后续动作：
  - 用较大 `num_envs` 或复用用户后续训练解析新增字段，确认大规模 reach 发散时极端 target 是否仍主要来自 open 样本。
  - 在改 reach clamp 前，先补一个离线/单元诊断：构造少量 non-done + 极端 `V_reach_append` 的 `calculate_reach_gae()` 用例，验证 bootstrap 污染路径。
  - 若来自 non-done bootstrap，优先给 reach bootstrap value/target 做语义边界约束。
  - 若 done 组也异常，优先检查 `calculate_reach_gae()` 的 done 处理和时间对齐。
  - 若 reach target 被控制后 `energy_min_ratio` 仍接近 1，下一步单独处理 energy scale、`min_energy` 和 earliest-index done 语义。
- 验证后下一步计划：
  - 如果短训中 `t_open_min` 明显比 `t_done_min` 更极端，下一轮实现 reach bootstrap/target clamp，并补单元测试。
  - 如果短训中 `t_done_min` 也进入异常大负值，下一轮写最小张量用例复现 `calculate_reach_gae()` 的 done 分支。
  - 如果两组 target 都可控但 `success` 仍坍塌，转向 energy 饱和和 combined advantage 语义。

### D003 补充：20260606-230145 证明 open bootstrap 污染路径

- 日期：2026-06-07
- 状态：open
- 严重性：P0
- 触发原因：用户新增 `open_ratio`、`t_done`、`t_open`、`tmin_src` 后重新跑了大批量训练。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260606-230145/training.log`
- 现象：
  - 共解析到 1286 条 iter 记录和 128 条 debug 记录。
  - `success` 峰值仍为约 0.307，最终约 0.013，说明 D002 的 std 限制没有解决主训练坍塌。
  - `std_mean` 在早期达到 1.0 后保持上界，说明探索噪声仍受控，不再是无界增长问题。
  - `reach_loss` 在 iter 24 超过 1e6，iter 36 超过 1e9，iter 43 超过 1e12，iter 335 超过 1e18，最大约 4.12e19。
  - `done_mean` 约 0.993-0.998，`energy_min_ratio` 约 0.972，energy/done 饱和仍稳定存在。
  - `open_ratio` 很低，约 0.0023-0.0072，但 `t_open_min` 最低达到约 -8.59e9。
  - `t_done_min` 始终为 -300，没有进入异常大负值。
  - `tmin_src done` 在所有 debug 行中均为 0，`tmin_t` 均为 199，`tmin_energy` 均为 -400。
- 初步假设：
  - H1：极端 `targets_reach` 由极少量 `done_for_gae == 0` 的 open 样本通过下一步 reach value bootstrap 引入。
  - H2：done 分支 target 保持在语义范围内，因此当前首要问题不是 done 分支的 `h/g` 原始信号异常。
  - H3：reach critic 输出发散后会反过来污染 open 样本 target，形成自激发闭环。
  - H4：energy/done 饱和仍是独立问题，但在修复 reach bootstrap 污染之前，难以判断其对策略学习的真实影响。
- 代码链路：
  - `EC_EFPPO_Buffer.compute_advantages()` 构造 `V_reach_append`、`V_total_append` 和 `done_for_gae`。
  - `calculate_indexs3()` 返回的 `done_for_gae` 使绝大多数样本走 done 分支，少量 open 样本保留 bootstrap。
  - `calculate_reach_gae()` 对 open 样本使用下一步 `V_reach`，当 reach critic 已发散时，target 被带到 1e9 量级。
- 证据：
  - `t_done_min=-300` 稳定，说明 done 分组 target 没有爆炸。
  - `tmin_src done=0` 全覆盖 debug 行，说明每次全局最小 target 都来自 open 样本。
  - `open_ratio` 极低但 `t_open_min` 极端，说明少量样本足以支配 MSE 形式的 reach loss。
  - `tmin_t=199` 全覆盖，说明问题集中在 rollout 末端 bootstrap，而不是任意中间步的原始 reach/cost 信号。
- 结论：
  - 当前最优先的改动应是约束 reach critic bootstrap value 的语义范围，先切断 open 样本 target 被发散 value 污染的闭环。
  - 该改动不是为了掩盖 loss，而是为了让 reach target 回到与 `h/g`、done 分支和任务代价同量级的可学习范围。
- 改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：为 `EC_EFPPO_Buffer` 增加 `reach_value_clip` 配置。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：在 `compute_advantages()` 中构造 `V_reach_for_bootstrap = clamp(V_reach_append, -clip, clip)`，并用于 `V_total_append`、`calculate_indexs3()` 和 `calculate_reach_gae()`。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：保留原始 `self.value_reach` 统计，同时新增 `reach_value_clip` 和 `reach_value_clip_ratio` 诊断字段。
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：设置 `reach_value_clip = 5000.0`。该值显著大于日志中 done 分支约 `[-300, 1365]` 的正常范围，但可以阻断 `-1e9` 级 bootstrap 污染。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：向 EC-EFPPO 传入 `reach_value_clip`，并在 debug 行输出 `reach_clip_ratio`。
  - `tests/test_ecfppo.py`：新增极端 `last_values_reach=-1e9` 的回归测试，验证 target 下界被裁剪保护。
  - `tests/test_train_ecfppo.py`：补充配置测试，确保默认 `reach_value_clip=5000.0`。
- 验证：
  - 已运行 `python3 -m py_compile rsl_rl/rsl_rl/algorithms/ecfppo.py legged_gym_go2/legged_gym/scripts/train_ecfppo.py legged_gym_go2/legged_gym/envs/go2/go2_config.py tests/test_ecfppo.py tests/test_train_ecfppo.py`。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo.py`，结果 17 passed。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_train_ecfppo.py'`，结果 12 passed。
- 后续动作：
  - 先跑 100-200 iter 大批量诊断训练，不建议直接跑完整 1500 iter。
  - 重点观察 `reach_clip_ratio` 是否非零但不过高、`t_open_min` 是否从 1e9 量级回到 5000 边界内、`reach_loss` 是否不再快速进入 1e12+。
  - 同时继续观察 `success`、`done_mean`、`energy_min_ratio`、`t_done_min` 和 `energy_loss`。
- 验证后下一步计划：
  - 若 `t_open_min` 和 `reach_loss` 明显受控且 success 不再早期坍塌，继续跑 300-500 iter 确认稳定性。
  - 若 `reach_clip_ratio` 长期很高，说明 reach critic 输出仍在大量越界，下一步检查 reach critic 学习率、value loss clipping 和 target 标准化。
  - 若 reach target 受控但 success 仍坍塌，下一步转向 energy/done 饱和：审查能耗尺度、`min_energy`、动作裁剪后能耗和 `calculate_indexs3()` earliest done 语义。
  - 若 `t_done_min` 也开始异常，下一步写最小张量用例检查 `calculate_reach_gae()` 的 done 分支时间对齐。

## 决策记录

- 2026-06-05：确认 `std` 未加入 policy optimizer 是确定实现 bug，已按最小修复处理。该改动不改变 EC-EFPPO 的 GAE/target 语义，只恢复参考实现中 policy 分布参数可训练的基本行为。
- 2026-06-05：新增诊断日志属于观测性改动，用于后续判断 reach critic 发散、energy 饱和和 `done_for_gae` 过密是否仍存在。
- 2026-06-05：临时训练已证明 `std` 修复生效，但没有证明训练稳定性已解决；下一轮应优先分析 energy 饱和和 done mask。
- 2026-06-06：确认直接优化实际 `std` 会造成探索噪声无界增长。EC-EFPPO 改为优化 `log_std`，通过 `exp(clamp(log_std))` 得到标准差，并降低/退火 entropy，先消除策略分布层面的不稳定来源。
- 2026-06-06：最新训练证明 `std` 已受控，但 reach target/bootstrap 发散与 energy/done 饱和仍导致训练坍塌。下一步只增加分组诊断统计，不改变训练语义。
- 2026-06-06：已增加 D003 分组诊断统计并通过单元测试；50 iter 小规模短训完成。短训未复现 reach loss 爆炸，但确认 energy/done 饱和稳定存在，且新增分组字段可用于后续大规模日志判定。
- 2026-06-07：20260606-230145 大批量日志证明极端 reach target 来自极少量 open/bootstrap 样本。已增加 `reach_value_clip=5000.0` 作为 bootstrap value 语义边界，并记录 `reach_clip_ratio` 用于下一轮诊断。

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
