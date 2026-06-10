# HDMCRA Debug Log

本文档用于记录 HDMCRA 当前阶段的训练稳定性诊断。旧的实现计划和历史 debug 记录已经不再作为当前判断依据；后续分析从本文档重新开始。

## 当前状态

- EC-EFPPO 的主体实现已经完成，可以运行端到端 Go2 训练。
- 当前尚未证明训练可以稳定达到 Reach-Avoid PPO 基线水平。
- 后续工作重点是基于训练日志和代码链路，判断算法语义、环境信号、超参数、网络结构或统计口径中是否存在不合理设计。
- 任何改动都需要先形成可验证假设，再通过测试或训练结果验证。

## 新手接手摘要

- 当前项目不是“实现未完成”，而是“实现已完成但训练稳定性尚未验证通过”。接手后不要先大改结构，也不要直接长训 1500 iter。
- 训练稳定性诊断已经从 D001 推进到 D012：先后处理了 `std`（动作标准差）未优化、`std` 无界增长、reach bootstrap target 发散、能耗尺度过大、动作饱和、reach critic 更新过强、actor mean 越界、policy advantage 符号方向、安全失败分组诊断、resume schedule 语义、bounded actor mean，以及 raw mean 饱和。
- 当前默认关键配置：`log_std_max=log(0.5)`、`policy_learning_rate=1e-4`、`energy_learning_rate=1e-3`、`reach_learning_rate=3e-4`、`reach_value_clip=5000.0`、`bounded_actor_mean=True`、`actor_raw_mean_bound=2.0`、`actor_raw_mean_bound_coef=1e-3`、`actor_mean_bound=1.0`、`actor_mean_bound_coef=1e-2`、`debug_stats_interval=10`。
- D008 已修正 `advantages_total`（组合优势）在 policy loss 中的符号方向：标准化后取负，使更小的 cost-like reach-avoid 值对应更高动作概率。
- 最新已完成 4096 env 短训 `20260609-174808` 显示 D008 后 `reach_rate`（到达目标比例）明显恢复，峰值 `success=0.322`、`reach_rate=0.688`；失败瓶颈从纯 `no_reach`（未到达）转向未到达和 `unsafe_before_reach`（到达前不安全）并存。
- D010 已修复 resume schedule 语义问题；`20260609-233704` 不能作为干净续训结论。
- D011 采用 `tanh(mean)` 后，干净训练 `20260610-133801` 证明无界越界已被消除，但 bounded mean 迅速贴到 `tanh` 边界，`act_mean_clip_ratio` 后期约 0.99，策略变成有界但饱和的 bang-bang policy。
- 当前进入 D012：增加 raw actor mean 诊断和正则，约束 `tanh` 前 logits 不要长期超过 `actor_raw_mean_bound=2.0`，先验证是否能降低 `raw_mean_clip_ratio/action_clip_ratio` 并恢复到达能力。

## 当前待分析问题

| ID | 状态 | 严重性 | 问题 | 相关日志/文件 | 下一步 | 验证后下一步计划 |
|---|---|---|---|---|---|---|
| D001 | resolved | P0 | 首轮全量训练 success 低平台，critic loss 量级异常，entropy/std 不更新 | `legged_gym_go2/logs/ecfppo_go2/20260605-102937/training.log` | 已修复 `std` 未进入 policy optimizer，并补充训练诊断指标 | 已被 D002 取代：`std` 能更新后出现无界增长，需改为 `log_std` 参数化并限制探索噪声 |
| D002 | open | P0 | `std` 可更新后无界增长，entropy 推高探索噪声，success 早期峰值后坍塌 | `legged_gym_go2/logs/ecfppo_go2/20260605-231206/training.log` | 已用 `log_std -> exp` 和上界控制探索噪声；最新训练证明 std 已受控但 success 仍坍塌 | 转入 D003：定位 reach target/bootstrap 发散与 energy/done 饱和的因果关系 |
| D003 | open | P0 | reach target/bootstrap 发散与 energy/done 饱和 | `legged_gym_go2/logs/ecfppo_go2/20260606-103051/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260606-230145/training.log` | 已确认极端 target 来自极少量 non-done/open bootstrap，已增加 reach bootstrap value 语义裁剪 | 跑 100-200 iter 大批量诊断训练，观察 `reach_clip_ratio`、`t_open_min`、`reach_loss` 和 success 是否改善；若仍失败，转向 energy/done 语义 |
| D004 | open | P0 | energy/done 饱和导致训练语义退化 | `legged_gym_go2/logs/ecfppo_go2/20260607-094159/training.log` | 增加 energy/action 诊断字段，不改变训练逻辑，跑 100-200 iter 短训 | 若 energy 很快掉到下界且动作大量裁剪，优先校准能耗尺度；若 energy 正常但 critic 仍越界，再处理 reach critic 学习率/输出约束 |
| D005 | open | P0 | 动作饱和导致策略-执行语义错配 | `legged_gym_go2/logs/ecfppo_go2/20260607-225408/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260608-113056/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260608-133653/training.log` | `policy_learning_rate=1e-4` 已显著改善动作贴边，但 success 未稳定，动作问题不再是唯一主因 | 转入 D006：优先处理 `reach critic` 输出越界和 `reach_clip_ratio` 偏高，不继续单纯降低 policy 学习率 |
| D006 | open | P0 | reach critic 输出越界导致动作改善后仍训练坍塌 | `legged_gym_go2/logs/ecfppo_go2/20260608-133653/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260608-151933/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260608-221634/training.log`、`rsl_rl/rsl_rl/algorithms/ecfppo.py` | 中等规模续训证明 `reach_learning_rate=3e-4` 只短期有效，后期 actor mean 再次发散，动作/能耗/critic 联合恶化 | 不继续长训；下一步优先处理 actor mean 边界约束或边界正则，同时保持 reach 降速配置 |
| D007 | open | P0 | actor mean 发散导致 raw action 与 clipped action 语义错配 | `legged_gym_go2/logs/ecfppo_go2/20260608-221634/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260609-080519/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260609-121135/training.log`、`legged_gym_go2/logs/ecfppo_go2/20260609-144319/training.log`、`rsl_rl/rsl_rl/algorithms/ecfppo.py`、`rsl_rl/rsl_rl/modules/actor_critic.py`、`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` | `actor_mean_bound_coef=1e-2` 进一步稳定 energy/reach critic 和动作幅度，但 success 仍在高峰后断崖坍塌；已补 success 分解和动作分量诊断；运行中短训早期显示第 0 维动作均值越界异常突出 | 完成 100-150 iter 诊断短训，重点判断 `no_reach` 是否主导失败，以及第 0 维动作是否长期越界；再决定调正则、改动作分布或检查动作语义 |
| D008 | resolved | P0 | EC-EFPPO policy advantage 符号方向与 reach-avoid cost-like 语义不一致 | `legged_gym_go2/logs/ecfppo_go2/20260609-144319/training.log`、`rsl_rl/rsl_rl/algorithms/ecfppo.py`、`rsl_rl/rsl_rl/algorithms/reach_avoid_ppo.py`、`legged_gym_go2/legged_gym/envs/go2/high_level_navigation_env.py` | 已确认旧 policy loss 会增大正 `advantages_total` 样本概率，但 `g/h` 定义和 Reach-Avoid PPO 基线均指向越小越好 | D008 后 `20260609-174808` 的 `reach_rate` 明显恢复，`no_reach` 不再单独主导；转入 D009 分析安全失败 |
| D009 | open | P0 | D008 后到达能力恢复但安全失败升高，需要分组诊断 | `legged_gym_go2/logs/ecfppo_go2/20260609-174808/training.log`、`legged_gym_go2/legged_gym/scripts/train_ecfppo.py` | 已新增 `succ/unsafe/noreach` 分组 debug 行，不改变训练语义 | 跑 100-200 iter 诊断短训，比较三组 `adv/hmax/align/act/mean_clip`，定位安全失败来自避障信号、动作饱和还是 advantage 区分不足 |
| D010 | resolved | P0 | resume 时 schedule 被新的 `max_iterations` 重算，导致 entropy/gamma 退火状态回退 | `legged_gym_go2/logs/ecfppo_go2/20260609-233704/training.log`、`legged_gym_go2/legged_gym/scripts/train_ecfppo.py`、`rsl_rl/rsl_rl/algorithms/ecfppo.py` | 已保存/恢复 `schedule_total_updates`，旧 checkpoint fallback 到 `start_iteration`，并将 entropy 退火下限 clamp 到 0 | 重新从头训练或重新 resume 时使用修复后的代码；正在运行的旧进程不会自动生效，需要重启 |
| D011 | resolved | P0 | 干净从头训练仍出现 actor mean 大面积越界，raw policy 与 clipped execution 错配 | `legged_gym_go2/logs/ecfppo_go2/20260610-080058/training.log`、`rsl_rl/rsl_rl/modules/actor_critic.py`、`legged_gym_go2/legged_gym/envs/go2/go2_config.py` | 已新增 `bounded_actor_mean=True`，用 `tanh(mean)` 将策略均值限制在 `[-1, 1]`，暂不改变 PPO log_prob 分布形式 | D011 消除了无界越界并降低采样动作裁剪，但 bounded mean 后期几乎全贴边；转入 D012 约束 raw mean 饱和 |
| D012 | open | P0 | bounded actor mean 变成 tanh 饱和，策略仍大量贴边且到达能力弱 | `legged_gym_go2/logs/ecfppo_go2/20260610-133801/training.log`、`rsl_rl/rsl_rl/modules/actor_critic.py`、`rsl_rl/rsl_rl/algorithms/ecfppo.py` | 新增 `raw_action_mean` 存储、debug 字段和 `actor_raw_mean_bound` 正则，默认 `bound=2.0`、`coef=1e-3` | 从头跑 200-300 iter，观察 `raw_mean_clip_ratio`、`act_mean_clip_ratio`、`action_clip_ratio`、`success/reach_rate/no_reach`；若 raw logits 仍饱和，再提高 raw 正则或降低 policy LR/std |

## 训练记录索引

| Run ID | 日期 | 命令/配置 | 迭代范围 | peak success | final success | 关键异常 | 结论 |
|---|---|---|---|---|---|---|---|
| 20260605-102937 | 2026-06-05 | `train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500` | 1-1500 | 0.194 | 0.171 | `reach_loss` 到 1e20 后仍停在 1e16；`energy_loss` 到 1e7；entropy 恒定 | 训练完整但未稳定收敛，存在明确实现/设计可疑点 |
| 20260605-231206 | 2026-06-05 | 修复 `std` optimizer 后再次训练 | 1-1387 | 0.339 | 约 0.096 | `std_mean` 从约 1.3 增至 100+；entropy 从约 4.3 增至 18+；`reach_loss` 延迟但仍到 1e19 | `std` optimizer 修复生效，但直接优化 std 会导致探索噪声失控，success 早期改善后坍塌 |
| 20260606-103051 | 2026-06-06 | `log_std` 上界、低 entropy 并退火后的训练 | 1-446 | 0.307 | 0.042 | `std` 在 iter 30 后稳定为 1.0；`done_mean≈0.993-0.998`；`energy_min_ratio≈0.972`；`reach_loss` 到 4.39e18 | 探索噪声已受控，但 reach target/bootstrap 发散和 energy/done 饱和仍主导训练失败 |
| 20260606-143037 | 2026-06-06 | `--headless --num_envs 16 --max_iterations 50`，D003 分组 debug | 1-50 | 0.375 | 0.125 | 小规模未复现 reach loss 爆炸；`done_mean=0.9625-0.9981`；`energy_min_ratio=0.9602-0.9701`；`std_mean` 最终 0.9962 | 分组 debug 可用；energy/done 饱和稳定存在，早期最小 target 可来自极少量 non-done 样本 |
| 20260606-230145 | 2026-06-06 | 大批量训练，包含 `open_ratio/t_done/t_open/tmin_src` 分组字段 | 1-1286 | 0.307 | 0.013 | `tmin_src done` 全为 0；`tmin_t` 全为 199；`t_done_min=-300` 稳定；`t_open_min` 最低约 -8.59e9；`open_ratio` 仅约 0.0023-0.0072 | 极端 reach target 不是 done 分支或原始 `g/h` 造成，而是极少量 non-done/open 样本从发散的 reach value bootstrap 进入 target |
| 20260607-094159 | 2026-06-07 | `reach_value_clip=5000.0` 后完整 1500 iter 大批量训练 | 1-1500 | 0.278 | 0.038 | `t_open_min` 被限制到 -5000；`reach_loss` 最大约 1.92e11，未再到 1e12+；`reach_clip_ratio` 后期约 0.35-0.68；`done_mean≈0.996`；`energy_min_ratio≈0.973` | bootstrap target 爆炸已被抑制，但 energy/done 饱和和 reach critic 大量越界仍导致训练坍塌 |
| 20260607-214857 | 2026-06-07 | 100 iter D004 诊断短训，新增 energy/action debug 字段 | 1-100 | 0.288 | 0.058 | `e_cons_mean` 平均约 116.16，最大恒为 120；`first_emin_step` 平均约 5.67；`act_clip_ratio` 平均约 0.953；`done_mean` 平均约 0.995；`energy_min_ratio` 平均约 0.972 | 能量约 5-6 个高层步触底，动作几乎全被裁剪，能耗尺度过大假设被直接验证 |
| 20260607-225408 | 2026-06-07 | `energy_consumption_scale≈0.5333` 后 100 iter 短训 | 1-100 | 0.260 | 0.047 | `e_cons_max` 已降到 8；`first_emin_step` 平均约 77.99；但 `act_clip_ratio` 平均约 0.954，最终约 0.998；`reach_loss` 最大约 3.01e8，最终约 1.33e8 | 能耗尺度修复有效，但动作仍几乎全贴边，reach critic 继续越界；下一步应优先降低动作饱和/探索强度 |
| 20260608-095413 | 2026-06-08 | `log_std_max=log(0.5)` 后 150 iter 短训，含动作均值诊断 | 1-150 | 0.314 | 0.029 | `std_mean≈0.5` 受控；`act_clip_ratio` 平均约 0.948；`act_mean_clip_ratio` 平均约 0.948；`act_mean_abs_mean` 后期到 5.89e3；`clipped_act_abs_mean` 平均约 0.974；`energy_loss` 最大约 9.20e4 | 动作贴边主要来自 actor mean 越界，不是探索噪声；下一步优先降低 policy 更新强度或约束 actor 输出均值 |
| 20260608-113056 | 2026-06-08 | `policy_learning_rate=3e-4`、critic LR 保持 `1e-3` 后 150 iter 短训 | 1-150 | 0.294 | 0.082 | `act_mean_abs_mean` 平均约 25.95，较上一轮 1477 大幅下降；但 `act_mean_clip_ratio` 平均仍约 0.897；`clipped_act_abs_mean` 平均约 0.948；`e_cons_mean` 平均约 7.45；后期 success 仍坍塌 | policy 降速有效但不足，actor mean 不再爆到千级但仍越界；下一步应继续处理动作均值约束或进一步降低 policy 更新强度 |
| 20260608-133653 | 2026-06-08 | `policy_learning_rate=1e-4`、critic LR 保持 `1e-3` 后 150 iter 短训 | 1-150 | 0.278 | 0.015 | `act_mean_clip_ratio` 平均约 0.690，`act_clip_ratio` 平均约 0.708，`clipped_act_abs_mean` 平均约 0.841，`e_cons_mean` 平均约 6.34，均较上一轮改善；但 `reach_clip_ratio` 平均升至约 0.381，`reach_loss` 平均约 3.62e7，success 后期坍塌更重 | policy 降到 `1e-4` 证明动作贴边可缓解，但训练失败主因转向 reach critic 输出越界；不应继续单纯降低 policy LR |
| 20260608-151933 | 2026-06-08 | `policy_learning_rate=1e-4`、`reach_learning_rate=3e-4` 后 150 iter 短训 | 1-150 | 0.177 | 0.110 | `reach_loss` 平均约 5.40e6，较上一轮 3.62e7 大幅下降；`reach_clip_ratio` 平均约 0.116，较上一轮 0.381 明显下降；但 `act_mean_clip_ratio` 平均约 0.744、最终约 0.909，`energy_loss` 平均约 1717、最终约 5227，动作贴边和 energy critic 后期回升 | reach 降速有效，训练不再早期完全坍塌；但动作饱和和 energy loss 后期回流，需要中等规模复测确认趋势 |
| 20260608-221634 | 2026-06-08 | 从 `20260608-151933/model_final.pt` 恢复，继续到 500 iter | 151-500 | 0.198 | 0.052 | `act_mean_clip_ratio` 平均约 0.934、后期约 0.973；`act_mean_abs_mean` 平均约 79.91，最大约 2063；`e_cons_mean` 平均约 7.65，`first_emin_step` 平均约 79.40；`energy_loss` 平均约 2.88e4，最大约 1.35e5；`reach_loss` 平均约 6.23e7，最大约 2.15e8 | 中等规模续训反证“继续训练会自然稳定”；短期 reach 降速有效，但长期 actor mean 发散和动作饱和重新主导训练失败 |
| 20260609-080519 | 2026-06-09 | `actor_mean_bound_coef=1e-3` 后 200 iter 短训 | 1-200 | 0.353 | 0.028 | `mean_bound_loss` 平均约 1.643；`act_mean_clip_ratio` 平均约 0.642，较无正则续训 0.934 明显下降；`act_mean_abs_mean` 平均约 3.43，较 79.91 大幅下降；但 151-200 窗口 `act_mean_clip_ratio` 回升到约 0.814，`e_cons_mean` 回升到约 7.01，`reach_loss` 平均约 1.14e8 | D007 方向有效但系数偏弱；actor mean 仍会后期重新贴边，下一步应提高正则系数而不是继续长训 |
| 20260609-121135 | 2026-06-09 | `actor_mean_bound_coef=1e-2` 后 200 iter 短训 | 1-200 | 0.289 | 0.011 | `energy_loss` 平均约 1.04e2，`reach_loss` 平均约 4.91e6，较 `1e-3` 明显下降；`act_mean_clip_ratio` 平均约 0.575，`act_mean_abs_mean` 平均约 1.86；但 iter 151-160 success 均值约 0.228 后，161-190 掉到约 0.012，且 critic 未同步爆炸 | 更强正则稳定了动作和 critic，但 success 仍存在策略/任务语义断崖；下一步应补 success 分解诊断，而不是立即继续加正则 |
| 20260609-144319 | 2026-06-09 | 新增 success 分解和动作分量诊断后的 200 iter 短训 | 1-200 | 0.289 | 0.011 | `161-200` 窗口 `no_reach≈0.981`、`unsafe_before_reach≈0.002`；最终 `act_mean_clip_ratio=0.7113`、`reach_clip_ratio=0.0235`，std 受控且 critic 未同步爆炸 | 失败主因是策略不去目标；结合代码链路转入 D008，优先检查 policy advantage 符号方向 |
| 20260609-165333 | 2026-06-09 | D008 符号修复后 `--num_envs 64 --max_iterations 50` 冒烟训练 | 1-50 | 0.203 | 0.016 | 后 10 iter `success≈0.072`、`reach_rate≈0.122`、`no_reach≈0.878`；最终 `act_mean_clip_ratio=0.4093`、`reach_clip_ratio=0.0020` | 小规模结果噪声大，不能证明收敛；但端到端链路正常，且后段 target-drive 有早期改善迹象，下一步需要 4096 env 100-150 iter 复测 |
| 20260609-174808 | 2026-06-09 | D008 符号修复后 4096 env 200 iter 诊断短训 | 1-200 | 0.322 | 0.322 | 后 10 iter `success≈0.223`、`reach_rate≈0.472`、`unsafe_before_reach≈0.249`、`no_reach≈0.528`；最终 `act_mean_clip_ratio=0.7301`、`reach_clip_ratio=0.6039` | D008 恢复目标驱动；新瓶颈转为未到达和安全失败并存，进入 D009 分组诊断 |
| 20260609-213244 | 2026-06-09 | D009 分组诊断后 `--num_envs 64 --max_iterations 10` 冒烟训练 | 1-10 | 0.016 | 0.016 | `group 00010` 正常输出；`succ r=0.016`、`unsafe r=0.000`、`noreach r=0.984` | 证明新增分组日志在真实训练链路可用；该小规模结果不用于收敛判断 |
| 20260609-233704 | 2026-06-10 | 从 `20260609-213418/model_200.pt` resume 到 `--max_iterations 1500`，旧 schedule 语义 | 201-1102 | 0.436 | 0.019 | iter 201 的 `ent_coef=0.00087`、`gamma_reach=0.999264`，相对原 iter 200 的 `ent_coef≈0.00001`/`gamma_reach=0.999990` 发生退火回退；后段 `no_reach≈0.98` | 不能作为干净续训结论；触发 D010 修复 resume schedule 语义 |
| 20260610-080058 | 2026-06-10 | D010 后从头训练，未启用 bounded mean | 1-500 | 0.289 | 0.052 | 后段 `act_mean_clip_ratio≈0.84`，iter 500 `reach_rate=0.060`、`no_reach=0.940`、`reach_clip_ratio=0.9262`；`succ/unsafe/noreach` 三组均有较高 mean clip | schedule 已干净，但动作均值越界仍未解决；触发 D011，用 bounded actor mean 先消除均值越界再复测 |
| 20260610-133801 | 2026-06-10 | D011 `bounded_actor_mean=True` 后从头训练，阶段性到 iter 446 | 1-446 | 0.115 | 0.058 | `act_clip_ratio` 后期约 0.50，低于上一轮 0.77-0.84；但 `act_mean_clip_ratio` 从 iter 200 起约 0.99，说明 bounded mean 被推到 `tanh` 边界；后段 `no_reach≈0.935` | D011 消除无界 raw action 爆炸但引入/暴露 tanh 饱和；触发 D012，对 raw mean 加正则和诊断 |

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

### D004: energy/done 饱和导致训练语义退化

- 日期：2026-06-07
- 状态：open
- 严重性：P0
- 触发原因：`reach_value_clip=5000.0` 后完整 1500 iter 大批量训练仍未稳定，success 后期坍塌。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260607-094159/training.log`
- 现象：
  - 共 1500 条 iter 记录和 150 条 debug 记录，训练完整保存 checkpoint。
  - `success` 峰值为 0.278（iter 24），最终为 0.038；101-200 iter 均值约 0.145，之后长期降到约 0.036。
  - `reach_loss` 最大约 1.92e11，最终约 1.11e11；相比上一轮 4.12e19 明显下降，且未再超过 1e12。
  - `t_open_min` 和 `t_reach_min` 被限制在 -5000 附近，说明 open bootstrap target 爆炸已被抑制。
  - `reach_clip_ratio` 后期常在 0.35-0.68，p50 约 0.4115，说明 reach critic 输出仍大量超过 `±5000` 语义边界。
  - `done_mean` p50 约 0.9966，`energy_min_ratio` p50 约 0.9727，说明绝大多数 rollout 时间步仍被 done/energy 下界主导。
- 初步假设：
  - H1：当前主要矛盾已从 bootstrap target 无界爆炸转移为 energy/done 饱和导致训练语义退化。
  - H2：高层能耗公式 `sum(clipped_action^2) * energy_consumption_scale * high_level_action_repeat` 在 Go2 三维动作和 repeat=5 下可能过大，导致能量很快掉到 `min_energy=-400`。
  - H3：大量样本被 done mask 截断后，只有极少 open 样本承担 bootstrap，策略和 critic 得到的有效学习信号不足。
  - H4：reach critic 大量越界仍需处理，但在 energy/done 饱和未诊断清楚前，不应继续调 `reach_value_clip`。
- 代码链路：
  - `HighLevelNavigationEnv.update_energy()`：按裁剪后的高层动作平方和乘以 `energy_consumption_scale` 与 `repeat` 扣能量。
  - `HierarchicalGO2Env.step()`：当前 `high_level_action_repeat=5`，每个高层动作驱动 5 次底层执行。
  - `EC_EFPPO_Buffer.compute_advantages()`：energy 序列进入 `calculate_indexs3()`，生成 `done_for_gae`，再影响 reach/energy/combined advantage。
  - `train_ecfppo.py`：目前日志缺少每步能耗、初始能量、首次到达能量下界步数和动作裁剪比例，无法直接判断能量饱和的具体触发路径。
- 证据：
  - D003 修改后 `t_open_min` 不再进入 1e9 量级，说明 bootstrap target 污染已被限制。
  - 但 `energy_min_ratio≈0.973` 与 `done_mean≈0.996` 基本没有改善，说明训练样本仍高度退化。
  - 当前能耗最大值按配置可达 `3 * 8 * 5 = 120` 每高层步，而初始能量只在 `[-400, 800]` 之间均匀采样，horizon 为 200。
- 结论：
  - 当前不应继续跑 1500 iter，也不应继续调 `reach_value_clip`。
  - 下一步应先增加 energy/action 观测字段，确认能量到底在第几步耗尽、每步耗能多大、动作是否大量被裁剪。
- 改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：新增 `energy_consumption_mean/max`、`first_energy_min_step_*`、`action_clip_ratio`、`action_abs_mean/max`、`init_energy_min/mean/max` 诊断字段。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：在 debug 行输出 `e_cons`、`first_emin_step`、`act_clip_ratio` 和 `init_energy`。
  - `tests/test_ecfppo.py`：新增确定性测试，验证能耗均值/最大值、初始能量统计、首次触底步数和动作裁剪比例。
  - 该改动只增加诊断统计，不改变训练逻辑。
- 验证：
  - 已运行 `python3 -m py_compile rsl_rl/rsl_rl/algorithms/ecfppo.py legged_gym_go2/legged_gym/scripts/train_ecfppo.py tests/test_ecfppo.py`。
  - 已运行 `conda run -n hdmcr python tests/test_ecfppo.py`，结果 18 passed。
  - 已完成 100 iter 诊断短训：`logs/ecfppo_go2/20260607-214857/training.log`。
  - 短训 `debug 00010` 显示 `e_cons [9.401e+01, 1.200e+02]`、`first_emin_step 6.68`、`act_clip_ratio 0.6824`、`init_energy [-3.996e+02, 1.989e+02, 7.992e+02]`。
  - 完整 10 条 debug 统计显示：`energy_consumption_mean` 平均约 116.16，`energy_consumption_max` 恒为 120，`first_energy_min_step` 平均约 5.67，`action_clip_ratio` 平均约 0.953。
  - 同一短训中 `done_mean` 平均约 0.995，`energy_min_ratio` 平均约 0.972，`open_ratio` 平均约 0.00484。
- 后续动作：
  - 已将 `energy_consumption_scale` 从默认 8.0 改为 `8.0 / (3 * 5) = 0.533333...`，用于把三维动作和 5 次低层重复归一化。
  - 下一步只跑 100-200 iter 短训，验证 `e_cons` 最大值是否从 120 降到约 8，`first_energy_min_step` 是否明显变晚，`energy_min_ratio` 和 `done_mean` 是否下降。
- 验证后下一步计划：
  - 若 energy 饱和明显改善但 `act_clip_ratio` 仍接近 1，下一步再降低 `log_std_max` 或初始探索噪声。
  - 若 energy 饱和改善且 success 不再坍塌，继续 300-500 iter 中等规模确认趋势。
  - 若 energy 仍很快掉到 -400，则继续审查初始能量分布和能耗公式，而不是调 `reach_value_clip`。
- 最新验证：
  - 已完成能耗缩放后的 100 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260607-225408/training.log`。
  - `e_cons_mean` 平均约 7.76，`e_cons_max` 恒为 8.0，说明能耗缩放配置已生效。
  - `first_energy_min_step` 平均约 77.99，相比缩放前约 5.67 明显变晚，说明“几步内触底”的问题已缓解。
  - `done_mean` 平均约 0.922，`energy_min_ratio` 平均约 0.612，明显低于缩放前约 0.995 和 0.972，但仍偏高。
  - `open_ratio` 平均约 0.078，高于缩放前约 0.00484，说明有效 non-done/open 样本变多。
  - `act_clip_ratio` 平均约 0.954，最终约 0.998，动作仍几乎全贴在边界，导致每步耗能仍接近最大值 8。
  - `reach_clip_ratio` 平均约 0.367，`reach_loss` 最大约 3.01e8、最终约 1.33e8，reach critic 仍明显越界和不稳定。
- 最新结论：
  - D004 中“能耗尺度过大”的子问题已被有效修复。
  - 当前剩余主矛盾变为：策略输出/探索导致动作饱和，进而保持高耗能；同时 reach critic 输出仍大量越过 `reach_value_clip` 语义边界。
  - 下一步不应继续扩大训练轮数，而应先降低动作饱和来源，例如降低 `log_std_max` 或初始探索噪声，并继续用 100 iter 短训验证。
- 最新验证后下一步计划：
  - 优先把 `log_std_max` 从 0 降到负值（例如 `log(0.5)≈-0.693`），让最大标准差从 1.0 降到 0.5，目标是把 `act_clip_ratio` 从约 0.95 明显压低。
  - 如果动作贴边下降后 `e_cons_mean` 低于 8 且 `done_mean/energy_min_ratio` 继续下降，再跑 300-500 iter 中等规模确认 success 是否不再早期坍塌。
  - 如果动作贴边下降但 `reach_clip_ratio/reach_loss` 仍高，再单独处理 reach critic 的输出约束、学习率或损失尺度。


### D005: 动作饱和导致策略-执行语义错配

- 日期：2026-06-08
- 状态：open
- 严重性：P0
- 触发原因：`energy_consumption_scale≈0.5333` 后，能耗尺度已修复，但动作仍几乎全部贴边。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260607-225408/training.log`
- 现象：
  - `e_cons_max` 已从 120 降到 8，说明 D004 的能耗尺度修复生效。
  - `first_energy_min_step` 平均约 77.99，明显晚于修复前约 5.67。
  - `done_mean` 平均约 0.922，`energy_min_ratio` 平均约 0.612，均明显低于修复前，但仍偏高。
  - `act_clip_ratio` 平均约 0.954，最终约 0.998，说明动作仍几乎全部被裁剪到 `[-1, 1]` 边界。
  - `debug 00010` 时 `std_mean≈0.415`，但 `act_clip_ratio≈0.814`，说明动作贴边不一定只由标准差上限过大造成，也可能来自 actor 输出均值快速越界。
  - `reach_loss` 最大约 3.01e8，最终约 1.33e8；`reach_clip_ratio` 平均约 0.367，reach critic 仍明显越界。
- 初步假设：
  - H1：当前主要矛盾从“能耗尺度过大”转移为“动作饱和”。
  - H2：由于环境执行前会裁剪动作，PPO 记录的原始采样动作与环境实际执行的裁剪动作可能语义错配。
  - H3：动作贴边可能来自探索噪声过大，也可能来自 actor mean 本身被策略更新推到边界外。
  - H4：在动作饱和未缓解前，直接处理 reach critic 可能会被劣质 rollout 数据干扰。
- 代码链路：
  - `EC_EFPPO_ActorCritic.update_distribution()`：actor 输出 `mean`，策略分布为 `Normal(mean, std)`。
  - `EC_EFPPO_ActorCritic.act()`：从未裁剪的 Normal 分布采样动作并计算 log probability。
  - `HighLevelNavigationEnv.update_velocity_commands()` 和 `update_energy()`：环境执行和能耗计算前都会把动作裁剪到 `[-1, 1]`。
  - `EC_EFPPO_Buffer.compute_advantages()`：已有 `action_clip_ratio` 只能说明原始动作是否贴边，不能区分是均值越界还是噪声采样越界。
- 改动：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：将 `log_std_max` 从 `0.0` 改为 `-0.6931471805599453`，即最大 `std` 从 1.0 降为 0.5。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：buffer 新增 `action_mean`，并记录 `action_abs`、`action_mean_abs`、`clipped_action_abs`、`action_clip_ratio` 和 `action_mean_clip_ratio`。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：debug 行新增 `act_abs`、`act_mean_abs`、`act_mean_clip_ratio`、`clipped_act_abs`。
  - `tests/test_ecfppo.py`：补充确定性测试，验证动作均值、原始动作和裁剪后动作统计。
  - `tests/test_train_ecfppo.py`：同步 `log_std_max` 配置断言。
- 验证：
  - 已运行 `python3 -m py_compile rsl_rl/rsl_rl/algorithms/ecfppo.py legged_gym_go2/legged_gym/scripts/train_ecfppo.py legged_gym_go2/legged_gym/envs/go2/go2_config.py tests/test_ecfppo.py tests/test_train_ecfppo.py`。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_ecfppo.py'`，结果 18 passed。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_train_ecfppo.py'`，结果 12 passed。
  - 已运行 `git diff --check`，无空白格式问题。
- 后续动作：
  - 跑 100 iter 短训，不要长训。
  - 目标 1：`std_mean` 不应超过 0.5。
  - 目标 2：`act_clip_ratio` 应明显低于 0.95；若仍高，查看 `action_mean_clip_ratio` 判断是否 actor mean 越界。
  - 目标 3：`clipped_act_abs_mean` 和 `e_cons_mean` 应低于满动作/满耗能状态。
- 验证后下一步计划：
  - 若 `action_mean_clip_ratio` 低但 `act_clip_ratio` 高，下一步继续降低探索噪声或初始 `init_noise_std`。
  - 若 `action_mean_clip_ratio` 高，下一步拆分 policy/critic 学习率，优先降低 policy learning rate。
  - 若动作贴边缓解但 `reach_loss/reach_clip_ratio` 仍高，再进入 reach critic 专项处理。

- 最新验证：
  - 已完成 `log_std_max=log(0.5)` 后 150 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260608-095413/training.log`。
  - `std_mean` 平均约 0.4999，最大为 0.5，说明标准差上限修改生效。
  - `act_clip_ratio` 平均约 0.948，最终约 0.985；动作贴边问题没有解决。
  - `act_mean_clip_ratio` 平均约 0.948，最终约 0.985，几乎等于 `act_clip_ratio`，说明贴边主要来自 actor mean 越界，而不是采样噪声。
  - `act_mean_abs_mean` 平均约 1477，最终约 5889；`act_mean_abs_max` 最大约 81970，actor 输出均值已经严重跑飞。
  - `clipped_act_abs_mean` 平均约 0.974，最终约 0.993，环境实际执行动作仍接近满动作。
  - `e_cons_mean` 平均约 7.72，最大恒为 8.0，能耗仍接近满耗能。
  - `success` 峰值为 0.314（iter 51），最终为 0.029，后期再次坍塌。
  - `energy_loss` 最大约 9.20e4、最终约 7.23e4，energy critic 在后期明显变差。
- 最新结论：
  - 继续降低 `log_std_max` 的收益有限，因为主因不是探索标准差，而是 actor mean 被策略更新推到极大值。
  - 下一步应优先处理 policy 更新强度或 actor 输出均值约束，而不是继续调 energy scale 或 reach value clip。
- 最新验证后下一步计划：
  - 首选方案：拆分 policy/energy/reach optimizer 学习率，先把 policy learning rate 从 1e-3 降到 3e-4，energy/reach 可暂时保留 1e-3，跑 100 iter 验证。
  - 观察目标：`act_mean_clip_ratio` 明显下降，`act_mean_abs_mean` 不再进入百/千级，`clipped_act_abs_mean` 低于 0.9，`e_cons_mean` 低于满耗能 8。
  - 若降低 policy learning rate 后 actor mean 仍跑飞，再考虑 actor 输出均值约束，例如对 mean 使用 tanh 边界或更严格的 action mean 正则。

- 第二轮改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：新增 `policy_learning_rate`、`energy_learning_rate`、`reach_learning_rate`，三个优化器分别使用对应学习率；未配置时回退到 `learning_rate`，保持旧调用兼容。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：训练入口传入三路学习率配置。
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：保留 `learning_rate=1e-3` 作为 critic fallback，新增 `policy_learning_rate=3e-4`、`energy_learning_rate=1e-3`、`reach_learning_rate=1e-3`。
  - `tests/test_train_ecfppo.py`：新增三路学习率配置和 optimizer param group 断言。
- 第二轮验证：
  - 已运行 `python3 -m py_compile rsl_rl/rsl_rl/algorithms/ecfppo.py legged_gym_go2/legged_gym/scripts/train_ecfppo.py legged_gym_go2/legged_gym/envs/go2/go2_config.py tests/test_train_ecfppo.py`。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_ecfppo.py'`，结果 18 passed。
  - 已运行 `conda run -n hdmcr bash -lc 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"; python tests/test_train_ecfppo.py'`，结果 12 passed。
  - 已运行 `git diff --check`，无空白格式问题。
- 第二轮验证后下一步计划：
  - 跑 100 iter 短训，检查 `act_mean_clip_ratio` 是否低于上一轮约 0.948，`act_mean_abs_mean` 是否不再快速进入百/千级。
  - 若 policy 降速有效，再跑 300-500 iter 中等规模确认 success 是否不再后期坍塌。
  - 若 policy 降速无效，再考虑 actor mean 结构约束或正则化。

- 第三轮验证：
  - 已完成 `policy_learning_rate=3e-4` 后 150 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260608-113056/training.log`。
  - `act_mean_abs_mean` 平均约 25.95，最终约 56.0；相比上一轮平均约 1477、最终约 5889，policy 降速显著抑制了 actor mean 爆炸。
  - `act_mean_abs_max` 最大约 441.7；相比上一轮最大约 81970，极端值也明显下降。
  - `act_mean_clip_ratio` 平均约 0.897，最终约 0.960，仍然很高，说明 actor mean 仍大面积越过 `[-1, 1]`。
  - `act_clip_ratio` 平均约 0.896，最终约 0.961，仍与 `act_mean_clip_ratio` 基本一致，动作贴边仍主要来自 actor mean。
  - `clipped_act_abs_mean` 平均约 0.948，最终约 0.980，环境执行动作仍接近满动作。
  - `e_cons_mean` 平均约 7.45，最终约 7.785，仍接近满耗能 8。
  - `success` 峰值为 0.294（iter 49），最终为 0.082；61-90 iter 均值约 0.243，但 121-150 iter 均值回落到约 0.059。
  - `reach_loss` 最大约 1.49e8，最终约 2.97e6；`reach_clip_ratio` 平均约 0.198，最终约 0.038。
  - `energy_loss` 最大约 9.19e4，最终约 4.16e4，仍偏高。
- 第三轮结论：
  - 降低 policy learning rate 是正确方向，但单独从 1e-3 降到 3e-4 还不足以解决动作贴边。
  - actor mean 已不再数千级爆炸，但仍长期输出大于动作边界的值，导致执行动作接近满动作、能耗仍接近上限。
  - 下一步应在“继续降低 policy 更新强度”和“增加 actor mean 边界/正则”之间选择。
- 第三轮验证后下一步计划：
  - 保守方案：继续把 `policy_learning_rate` 从 3e-4 降到 1e-4，保持其他参数不变，跑 100-150 iter，验证 `act_mean_clip_ratio` 是否明显下降。
  - 更直接方案：给 actor 输出均值加边界约束，例如 `mean = tanh(raw_mean)`，但这属于策略分布语义改动，风险高于单纯调学习率。
  - 建议先做保守方案；若 1e-4 后仍贴边，再进入 actor mean 结构约束。

- 第四轮改动：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：将 `policy_learning_rate` 从 `3e-4` 降到 `1e-4`。
  - 其他关键配置保持不变：`energy_learning_rate=1e-3`、`reach_learning_rate=1e-3`、`log_std_max=log(0.5)`、`reach_value_clip=5000.0`、Go2 高层最大单步耗能约 8。
  - 本轮目标不是直接追求最终 success，而是验证更慢的 policy 更新是否能继续降低 actor mean 越界和执行动作贴边。
- 第四轮验证后下一步计划：
  - 跑 100-150 iter 短训，重点比较 `act_mean_clip_ratio`（动作均值越界比例）、`act_clip_ratio`（采样动作裁剪比例）、`clipped_act_abs_mean`（裁剪后执行动作绝对值均值）和 `e_cons_mean`（平均每步耗能）。
  - 若这些指标明显下降且 `success` 不再在 100 iter 后坍塌，再考虑 300-500 iter 中等规模训练。
  - 若 `policy_learning_rate=1e-4` 后动作仍长期贴边，则进入 actor mean 结构约束或正则化，不再继续单纯降低学习率。

- 第四轮验证：
  - 已完成 `policy_learning_rate=1e-4` 后 150 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260608-133653/training.log`。
  - `success` 峰值为 0.278（iter 38），最终为 0.015；1-30 iter 均值约 0.0867，31-60 iter 均值约 0.0911，61-90 iter 均值约 0.0345，91-120 iter 均值约 0.0479，121-150 iter 均值约 0.0351。
  - `act_mean_clip_ratio` 平均约 0.690，较上一轮约 0.897 明显下降；最终约 0.884，说明后期仍有回升。
  - `act_clip_ratio` 平均约 0.708，较上一轮约 0.896 明显下降。
  - `clipped_act_abs_mean` 平均约 0.841，较上一轮约 0.948 下降，环境执行动作不再全程接近满动作。
  - `e_cons_mean` 平均约 6.34，较上一轮约 7.45 下降；`first_emin_step` 平均约 96.0，较上一轮约 81.6 推迟。
  - `energy_loss` 平均约 104，较上一轮约 10451 大幅下降，说明动作和能耗链路确实改善。
  - `reach_clip_ratio` 平均约 0.381，较上一轮约 0.198 升高；`reach_loss` 平均约 3.62e7，较上一轮约 1.86e7 更差。
- 第四轮结论：
  - 降低 policy 学习率到 `1e-4` 对动作贴边有效，但没有解决训练坍塌，且 success 更早、更深地回落。
  - 当前不应继续降低 policy 学习率；继续降低会进一步削弱策略学习速度，却不能解释 `reach_clip_ratio` 和 `reach_loss` 恶化。
  - 现阶段主矛盾转向 reach critic 输出越界：target 已通过 bootstrap clip 控制在约 `[-5000, 1000+]`，但 `values_reach` 仍可到 `-6e4` 量级，critic 输出本身缺少足够约束。
- 第四轮验证后下一步计划：
  - 进入 D006，优先做 reach critic 稳定性改动；首选最小实验是降低 `reach_learning_rate`，例如从 `1e-3` 降到 `3e-4`，保持 `policy_learning_rate=1e-4` 和其他配置不变。
  - 若降低 reach 学习率仍无法降低 `reach_clip_ratio`，再考虑增加 reach critic 专用梯度裁剪或输出语义约束。

### D006: reach critic 输出越界导致动作改善后仍训练坍塌

- 日期：2026-06-08
- 状态：open
- 严重性：P0
- 触发原因：`policy_learning_rate=1e-4` 明显改善动作和能耗指标，但 success 仍坍塌，且 reach critic 输出越界比例升高。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260608-133653/training.log`
- 现象：
  - `act_mean_clip_ratio`、`act_clip_ratio`、`clipped_act_abs_mean` 和 `e_cons_mean` 均较上一轮改善，说明 D005 的 policy 降速方向有效。
  - `energy_loss` 从上一轮平均约 10451 降到约 104，说明能耗 critic 不再是当前最强异常。
  - `reach_clip_ratio` 从上一轮平均约 0.198 升到约 0.381，说明 reach value 输出越过 `reach_value_clip=5000` 的比例更高。
  - `reach_loss` 平均约 3.62e7，最大约 1.65e8，仍维持高量级。
  - `success` 峰值在 iter 38 后回落，最终仅 0.015。
- 初步假设：
  - H1：当前 reach bootstrap target 已被裁剪，但 reach critic 输出本身没有边界，仍会产生远超语义范围的 value。
  - H2：reach critic 使用 `reach_learning_rate=1e-3` 和通用 `max_grad_norm=0.5`，更新强度可能偏大；energy critic 已使用更严格的梯度裁剪后表现明显更稳定。
  - H3：当 reach critic 输出越界时，combined advantage 的策略信号会继续噪声化，即使动作贴边缓解，success 也会坍塌。
- 结论：
  - 下一步不应继续单纯降低 policy 学习率，也不应直接长训。
  - 应先用最小改动降低 reach critic 更新强度，验证 `reach_clip_ratio` 和 `reach_loss` 是否下降。
- 改动：
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：将 `reach_learning_rate` 从 `1e-3` 降到 `3e-4`。
  - 其他关键配置保持不变：`policy_learning_rate=1e-4`、`energy_learning_rate=1e-3`、`log_std_max=log(0.5)`、`reach_value_clip=5000.0`、Go2 高层最大单步耗能约 8。
  - `tests/test_train_ecfppo.py`：同步默认配置和 optimizer param group 的 reach 学习率断言。
- 后续动作：
  - 跑 100-150 iter 短训。
  - 重点观察 `reach_clip_ratio`（reach value 越界比例）、`reach_loss`（可达价值损失）、`act_mean_clip_ratio`（动作均值越界比例）和 `success`（成功率）。
- 验证后下一步计划：
  - 若 `reach_clip_ratio` 明显下降且 success 不再早期坍塌，再考虑 300-500 iter 中等规模训练。
  - 若 `reach_clip_ratio` 仍高，下一步增加 `max_grad_norm_reach` 或给 reach critic 输出增加语义边界/正则。

- 第一轮验证：
  - 已完成 `reach_learning_rate=3e-4` 后 150 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260608-151933/training.log`。
  - `reach_loss` 平均约 5.40e6，较上一轮约 3.62e7 大幅下降；最大约 2.77e7，较上一轮约 1.65e8 下降。
  - `reach_clip_ratio` 平均约 0.116，较上一轮约 0.381 明显下降；说明降低 reach critic 学习率有效缓解了 value 越界。
  - `success` 峰值为 0.177（iter 140），最终为 0.110；121-150 iter 均值约 0.125，明显好于上一轮的约 0.035。
  - `act_mean_clip_ratio` 平均约 0.744，最终约 0.909；`act_clip_ratio` 平均约 0.755，最终约 0.910，动作贴边后期重新变重。
  - `e_cons_mean` 平均约 6.62，最终约 7.52；`energy_loss` 平均约 1717，最终约 5227，energy critic 后期明显变差。
- 第一轮结论：
  - D006 的最小改动有效，reach critic 输出越界和 reach loss 被明显压低。
  - 这是近期第一次出现 150 iter 后段 success 均值高于前段、且 final success 未坍塌到极低值的训练。
  - 但动作饱和和 energy critic 异常后期回流，说明 D005/D004 的问题仍未彻底解决。
- 第一轮验证后下一步计划：
  - 暂不继续降低 `reach_learning_rate`，也不立刻引入 actor mean 结构约束。
  - 先跑 300-500 iter 中等规模复测，确认 success 后段改善是否能保持或继续上升。
  - 若中训中 `act_mean_clip_ratio` 长期接近 0.9、`e_cons_mean` 接近 8 或 `energy_loss` 继续升高，再进入 actor mean 正则/边界约束或 energy critic 更新强度调整。

- 第二轮验证：
  - 已从 `legged_gym_go2/logs/ecfppo_go2/20260608-151933/model_final.pt` 恢复并继续训练到 total iter 500：`legged_gym_go2/logs/ecfppo_go2/20260608-221634/training.log`。
  - 日志包含 iter 151-500 共 350 条迭代记录，确认 resume 生效。
  - `success` 平均约 0.0494，峰值约 0.198（iter 381），最终约 0.052；分段上 301-350 均值约 0.0166，451-500 均值约 0.0350。
  - 相比 `20260608-151933` 的 1-150 iter，`energy_loss` 平均从约 1.72e3 升到约 2.88e4，最大到约 1.35e5。
  - 相比 `20260608-151933`，`reach_loss` 平均从约 5.40e6 升到约 6.23e7，最大到约 2.15e8。
  - `act_mean_clip_ratio` 平均约 0.934，451-500 窗口平均约 0.973，说明 actor mean 绝大多数时间已经跑出动作边界。
  - `act_mean_abs_mean` 平均约 79.91，最大观测约 2063；执行端 `clipped_act_abs_mean` 平均约 0.967，说明真实执行动作长期贴近 `[-1, 1]` 边界。
  - `e_cons_mean` 平均约 7.65，接近当前高层单步最大耗能 8；`first_emin_step` 平均约 79.40，说明能量仍会在 rollout 中段前后触底。
  - `reach_clip_ratio` 平均约 0.550，在 351-400 窗口平均约 0.951，说明 reach critic 后期重新大量越过语义裁剪边界。
- 第二轮结论：
  - 中等规模续训没有延续 150 iter 短训的改善，反而暴露出长期不稳定。
  - `reach_learning_rate=3e-4` 对早期 reach critic 稳定有效，但不能单独解决训练稳定性。
  - 当前最强因果链更像是 actor mean 发散 -> 动作执行端长期被裁剪 -> 能耗接近上限 -> energy/done 信号退化 -> energy/reach critic 再次恶化。
- 第二轮验证后下一步计划：
  - 不继续从当前模型长训，也不优先继续降低 `reach_learning_rate`。
  - 保持 `policy_learning_rate=1e-4`、`reach_learning_rate=3e-4`，下一步做最小 actor mean 边界处理。
  - 优先方案：在 policy loss 中增加 actor mean 边界正则，例如惩罚 `relu(abs(mean)-1)^2`，并记录 `mean_bound_loss`；该方案比直接 `tanh(mean)` 更温和，不立即改变高斯策略分布语义。
  - 备选方案：直接对 actor mean 做 `tanh` 边界化，能更强地避免越界，但会更明显改变策略分布语义，应放在正则方案无效之后。
  - 修改后只跑 100-150 iter 短训，重点观察 `act_mean_clip_ratio`、`act_mean_abs_mean`、`clipped_act_abs_mean`、`e_cons_mean`、`energy_loss`、`reach_clip_ratio` 和 `success`。


### D007: actor mean 发散导致 raw action 与 clipped action 语义错配

- 日期：2026-06-09
- 状态：open
- 严重性：P0
- 触发原因：`20260608-221634` 续训到 total iter 500 后，`act_mean_clip_ratio` 后期约 0.97，`act_mean_abs_mean` 平均约 79.91，执行端 `clipped_act_abs_mean` 平均约 0.967。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260608-221634/training.log`
- 现象：
  - actor 输出的 raw mean 长期远超 `[-1, 1]`，但环境速度命令和能耗计算都会先将动作裁剪到 `[-1, 1]`。
  - PPO 的 log probability 仍基于 raw sampled action 计算，环境反馈却来自 clipped action，形成策略更新语义和执行语义错配。
  - `e_cons_mean` 接近当前高层单步最大耗能 8，`energy_loss` 和 `reach_loss` 后期重新升高。
- 代码链路：
  - `EC_EFPPO_ActorCritic.update_distribution()` 使用无边界 actor mean 构造 `Normal(mean, std)`。
  - `EC_EFPPO.update()` 使用 raw action 的 log probability 计算 PPO ratio。
  - `HighLevelNavigationEnv.update_velocity_commands()` 和 `update_energy()` 都先裁剪 high-level action，再用于速度命令和能耗。
- 结论：
  - 当前不应继续从 `20260608-221634` 长训，也不应优先继续单独降低 reach critic 学习率。
  - 需要先阻止 actor mean 无约束跑出环境执行动作边界，否则 energy/reach critic 诊断会持续被动作贴边污染。
- 改动：
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：新增 `actor_mean_bound` 和 `actor_mean_bound_coef`，policy loss 增加 `mean_bound_loss = mean(relu(abs(mean)-bound)^2)`。
  - `legged_gym_go2/legged_gym/envs/go2/go2_config.py`：默认 `actor_mean_bound=1.0`、`actor_mean_bound_coef=1e-3`。
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：iter 日志新增 `mean_bound_loss`。
  - `tests/test_ecfppo.py`、`tests/test_train_ecfppo.py`：补充配置、loss 字段和边界正则单元测试。
- 后续动作：
  - 跑 100-150 iter 短训，不从已发散的 500 iter checkpoint 继续。
  - 重点观察 `mean_bound_loss`、`act_mean_clip_ratio`、`act_mean_abs_mean`、`clipped_act_abs_mean`、`e_cons_mean`、`energy_loss`、`reach_clip_ratio`、`success`。
- 验证后下一步计划：
  - 若动作均值越界明显下降且 success 不再坍塌，再跑 300-500 iter 中等规模验证。
  - 若动作仍长期贴边，增大 `actor_mean_bound_coef` 或考虑 tanh/squashed Gaussian 方案。
  - 若动作问题缓解但 success 仍低，再回到 energy/reach critic 目标和学习率诊断。

- 第一轮验证：
  - 已完成 `actor_mean_bound_coef=1e-3` 后 200 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260609-080519/training.log`。
  - `success` 平均约 0.0681，峰值约 0.353（iter 72），最终约 0.028；说明早期探索/到达能力增强，但后期仍明显坍塌。
  - 相比无正则的 `20260608-221634`，`act_mean_clip_ratio` 平均从约 0.934 降到约 0.642，`act_mean_abs_mean` 平均从约 79.91 降到约 3.43，说明 actor mean 边界正则方向有效。
  - `clipped_act_abs_mean` 平均从约 0.967 降到约 0.820，`e_cons_mean` 平均从约 7.65 降到约 6.11，`energy_loss` 平均从约 2.88e4 降到约 1.15e3，说明动作饱和缓解确实降低了能耗和 energy critic 压力。
  - 但 151-200 窗口 `act_mean_clip_ratio` 回升到约 0.814，`clipped_act_abs_mean` 回升到约 0.907，`e_cons_mean` 回升到约 7.01，说明当前正则系数不足以长期压住动作贴边。
  - `reach_loss` 平均约 5.35e7，151-200 窗口平均约 1.14e8；`reach_clip_ratio` 后期仍约 0.446，reach critic 仍被后期动作/能耗退化牵动。
  - `mean_bound_loss` 平均约 1.643，最大约 16.986；在 `actor_mean_bound_coef=1e-3` 下，实际加到 policy loss 的量级只有约 0.0016 均值，明显小于 `actor_loss` 约 0.314。
- 第一轮结论：
  - D007 的方向被验证：它显著降低了 actor mean 发散和 energy loss。
  - 当前失败不是因为正则方向错误，而是正则权重太弱，后期动作越界重新主导训练。
- 第一轮验证后下一步计划：
  - 不继续长训当前 run。
  - 已将 `actor_mean_bound_coef` 从 `1e-3` 提高到 `1e-2`，保持其他配置不变，下一步跑 100-150 iter 单变量短训。
  - 若 100-150 iter 后 `act_mean_clip_ratio` 仍高于约 0.6，再考虑 `3e-2` 或改为 tanh/squashed Gaussian 方案。

- 第二轮验证：
  - 已完成 `actor_mean_bound_coef=1e-2` 后 200 iter 短训：`legged_gym_go2/logs/ecfppo_go2/20260609-121135/training.log`。
  - 相比 `1e-3`，`energy_loss` 平均从约 1.15e3 降到约 1.04e2，`reach_loss` 平均从约 5.35e7 降到约 4.91e6，`reach_clip_ratio` 平均从约 0.278 降到约 0.095。
  - `act_mean_clip_ratio` 平均从约 0.642 降到约 0.575，`act_mean_abs_mean` 平均从约 3.43 降到约 1.86，`clipped_act_abs_mean` 平均从约 0.820 降到约 0.786。
  - 151-160 窗口表现最好：`success` 均值约 0.228，峰值约 0.289，`act_mean_clip_ratio` 约 0.440，`reach_clip_ratio` 约 0.002，说明当前配置可以短暂进入较好策略区间。
  - 但 161-190 窗口 `success` 均值约 0.012，且对应 `reach_clip_ratio` 仍低、`energy_loss` 未爆炸，说明坍塌不再是旧的 critic 爆炸/动作极端饱和模式。
- 第二轮结论：
  - `actor_mean_bound_coef=1e-2` 是比 `1e-3` 更稳定的设置，动作、能耗、reach critic 都明显改善。
  - 但 success 断崖仍存在，下一步需要分解 success 统计，定位是未到达目标、提前不安全、还是 energy/done 语义造成训练信号错配。
  - 目前不建议直接继续提高到 `3e-2`，否则可能把动作进一步压小，但不能回答 success 断崖原因。
- 第二轮验证后下一步计划：
  - 保持 `actor_mean_bound_coef=1e-2` 不变。
  - 在 `compute_reach_avoid_success_rate()` 或训练日志中增加只读诊断字段：`reach_rate`（到达比例）、`safe_rate`（到达前安全比例）、`unsafe_before_reach_rate`（到达前不安全比例）、`no_reach_rate`（未到达比例）。
  - 同时增加三维高层动作的分量统计，例如 `act_mean_abs_dim` 和 `act_mean_clip_dim`，判断是否某个动作维度主导策略坍塌。
  - 跑 100-150 iter 诊断短训后，再决定是调正则系数、降低策略更新强度，还是进入 tanh/squashed Gaussian 方案。

- 第三轮诊断改动：
  - `legged_gym_go2/legged_gym/scripts/train_ecfppo.py`：新增 `compute_reach_avoid_metrics()`，保留旧 `compute_reach_avoid_success_rate()` 兼容接口；iter 日志新增 `reach_rate`（到达目标比例）、`safe_rate`（到达前安全比例）、`unsafe_before_reach`（到达前不安全比例）和 `no_reach`（未到达比例）。
  - `rsl_rl/rsl_rl/algorithms/ecfppo.py`：在 buffer debug 统计中新增动作分量字段，记录 `act_mean_abs_dim`（各动作维度均值绝对值）、`act_mean_clip_dim`（各动作维度均值越界比例）和 `clipped_act_abs_dim`（各动作维度实际执行动作绝对值）。
  - 该改动只增加诊断字段，不改变 rollout、GAE、loss、优化器和环境 step 逻辑。
- 第三轮改动后下一步计划：
  - 跑 100-150 iter 诊断短训。
  - 若 `no_reach` 高，优先判断动作是否被正则压得过小或目标驱动不足。
  - 若 `unsafe_before_reach` 高，优先检查避障/安全约束和高层动作方向。
  - 若某个 `act_mean_clip_dim` 长期显著高于其他维度，优先针对该动作维度检查动作语义、归一化和代价尺度。
- 运行中短训早期快照：
  - 日志：`legged_gym_go2/logs/ecfppo_go2/20260609-144319/training.log`。
  - 文档更新时训练尚未完成，早期数据只能作为诊断方向，不能作为最终结论。
  - 前 10-16 iter 中 `success` 仍低，`no_reach` 长期约 0.93-0.99，说明大部分环境还没有到达目标。
  - `unsafe_before_reach` 早期通常低于 `no_reach`，说明当前早期失败更像“不去/到不了目标”，而不是“大量到达前不安全”。
  - `debug 00010` 中 `reach_clip_ratio=0`、`std_mean≈0.497`，说明旧的 reach value 爆炸和 std 失控不是当前早期主导异常。
  - `debug 00010` 中 `act_mean_clip_dim=[0.9784, 0.0013, 0.0000]`，第 0 维动作均值几乎全部越界，而第 1/2 维基本正常；这是下一步最重要的新线索。
- 运行中短训完成后的判断分支：
  - 若第 0 维 `act_mean_clip_dim` 持续接近 1，同时 `no_reach` 高，优先检查第 0 维动作 `[vx]` 的命令方向、尺度、裁剪、目标投影和能耗/正则压力。
  - 若第 0 维只在早期越界、后期恢复，但 `no_reach` 仍高，优先检查 reach advantage 是否对目标接近提供有效梯度。
  - 若 `unsafe_before_reach` 后期升高，说明策略开始接近目标但安全失败，应转向 `h` 值、安全约束和避障行为诊断。
  - 若 `reach_clip_ratio` 后期重新升高，再回到 reach critic 输出约束、学习率或梯度裁剪诊断。

### D008: EC-EFPPO policy advantage 符号方向与 cost-like 语义不一致

- 日期：2026-06-09
- 状态：open
- 严重性：P0
- 触发原因：`20260609-144319` 完整 200 iter 短训在 `success` 短暂升高后断崖坍塌，失败由 `no_reach` 主导，critic 和 std 未同步爆炸。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260609-144319/training.log`
- 现象：峰值 `success=0.289` 出现在 iter 157，最终 `success=0.011`；`161-200` 窗口 `reach_rate≈0.019`、`unsafe_before_reach≈0.002`、`no_reach≈0.981`。
- 初步假设：`advantages_total` 来自 `g_append=max(reach, -energy)`，语义是越小越好；旧 policy loss 按 reward-max PPO 方向增大正 advantage 样本概率，可能把策略推向更差的 reach-avoid/cost 方向。
- 代码链路：`EC_EFPPO_Buffer.compute_advantages()` 构造 `g_append` 和 `advantages_total`；`EC_EFPPO.update()` 将 `advantages_total` 归一化后直接用于 PPO policy loss；`HighLevelNavigationEnv._compute_g_function()` 中目标内 `g=-300`、目标外更大，`_compute_h_function()` 中安全 `h=-300`、不安全 `h=300`。
- 证据：小张量诊断显示旧 loss 对 `advantages_total=[-1, 1]` 的梯度会降低低值样本 log-prob、增大高值样本 log-prob；Reach-Avoid PPO 基线在 policy loss 前显式执行 `gae_batch=-adv_batch`。
- 结论：旧 EC-EFPPO policy loss 符号方向与当前 reach-avoid cost-like 信号不一致，是 `no_reach` 主导坍塌的高优先级可验证原因。
- 改动：`rsl_rl/rsl_rl/algorithms/ecfppo.py` 新增 `_policy_gae_from_advantages()`，对 `advantages_total` 标准化后取负；`tests/test_ecfppo.py` 新增梯度方向测试，要求更小的 cost-like advantage 提高对应动作概率。
- 验证：`tests/test_ecfppo.py` 20/20 通过，`tests/test_ecfppo_gae.py` 11/11 通过，`tests/test_energy_state.py` 12/12 通过；D008 符号修复后 `20260609-165333` 50 iter 冒烟训练完成，峰值 `success=0.203`，后 10 iter `reach_rate≈0.122`、`no_reach≈0.878`。
- 后续动作：小规模冒烟训练噪声大，不能证明收敛；下一步用 4096 env 跑 100-150 iter 诊断短训。如果 `reach_rate` 持续提升且 `no_reach` 下降，再观察 `unsafe_before_reach` 是否成为新瓶颈；如果仍不改善，回到动作语义和 rollout 分组 advantage 统计。


### D009: D008 后安全失败升高的分组诊断

- 日期：2026-06-09
- 状态：open
- 严重性：P0
- 触发原因：D008 符号修复后的 4096 env 训练 `20260609-174808` 显示 `reach_rate` 明显恢复，但 `unsafe_before_reach` 升高，训练瓶颈从“不去目标”转为未到达和安全失败并存。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260609-174808/training.log`
- 现象：该 run 跑满 200 iter，峰值 `success=0.322`、`reach_rate=0.688`、`safe_rate=0.354`；后 10 iter 平均 `success≈0.223`、`reach_rate≈0.472`、`unsafe_before_reach≈0.249`、`no_reach≈0.528`。
- 初步假设：D008 已恢复目标驱动，但当前日志还不能判断安全失败来自 `h`（安全约束）信号不足、动作饱和/转向维度失控、还是 `advantages_total` 对安全成功和不安全到达区分不够。
- 代码链路：`compute_reach_avoid_metrics()` 生成 success/failure mask；`EC_EFPPO_Buffer` 保存 rollout 的 `advantages_total`、`g_values/h_values`、`actions/action_mean` 和 observations；`train_ecfppo.py` 在 debug interval 写训练诊断。
- 证据：D008 前 `20260609-144319` 后段 `no_reach≈0.981`；D008 后 `20260609-174808` 后 10 iter `reach_rate≈0.472`，说明策略开始接近目标；但 `unsafe_before_reach≈0.249`，说明安全失败成为新增主要瓶颈。
- 结论：下一步不应先调超参数或修改安全约束，而应先把 rollout 按 `succ/unsafe/noreach` 分组，直接比较三类轨迹的 advantage、`g/h`、目标方向对齐和动作越界。
- 改动：`train_ecfppo.py` 新增 `success_mask`、`unsafe_before_reach_mask`、`no_reach_mask`、`first_indices`；新增 `compute_rollout_group_debug_stats()` 和 `group` 日志行；`tests/test_train_ecfppo.py` 新增分组 mask 和分组统计测试。该改动只增加诊断，不改变 rollout、GAE、loss 或 optimizer。
- 验证：`tests/test_train_ecfppo.py` 15/15 通过；D009 冒烟训练 `20260609-213244` 跑满 10 iter，`training.log` 已写出 `group 00010`，包含 `succ/unsafe/noreach` 三组统计。
- 后续动作：跑 100-200 iter 诊断短训，读取每个 `group` 行，重点比较 `unsafe` 组相对 `succ` 组的 `hmax`、`align`、`act` 和 `mean_clip`；如果 `unsafe` 组目标对齐高但 `hmax` 为正，优先处理安全约束/避障信号；如果 `mean_clip` 高，优先处理动作边界和维度语义；如果 `adv` 无法区分三组，回到 combined advantage 构造。


### D010: resume 续训时 schedule 被新的 max_iterations 重算

- 日期：2026-06-10
- 状态：resolved
- 严重性：P0
- 触发原因：`20260609-233704` 从 `20260609-213418/model_200.pt` 续训到 `--max_iterations 1500` 后，iter 201 的 `ent_coef` 和 `gamma_reach` 不再延续原 checkpoint 末尾状态。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260609-233704/training.log`
- 现象：原 run iter 200 已经 `ent_coef≈0.00001`、`gamma_reach=0.999990`；续训 run iter 201 变为 `ent_coef=0.00087`、`gamma_reach=0.999264`。这相当于重新打开探索并改变 reach 折扣时间尺度。
- 初步假设：`train_ecfppo.py` 使用当前命令行的 `max_iterations` 作为 `total_updates`，而不是 checkpoint 原始 schedule horizon；旧 checkpoint 又没有保存 schedule horizon，导致 resume 时退火进度被新实验长度重算。
- 代码链路：`train_ecfppo.py` 读取 `start_iteration` 后设置 `total_updates=max_iterations`；每轮通过 `EC_EFPPO.compute_gamma_reach()` 和 `compute_entropy_coef()` 计算退火值；checkpoint 旧格式只保存 `iteration`，未保存 `schedule_total_updates`。
- 证据：`20260609-233704` 的 iter 201 日志直接显示 schedule 回退；同一长训后期出现动作饱和、critic 退化和 `no_reach` 重新主导，但该结果混入了 schedule 回退干扰，不能作为干净续训结论。
- 结论：resume 语义存在实现问题。继续分析长训前，应先修复 schedule 继承，否则从 checkpoint 续训和从头训练不可比。
- 改动：`train_ecfppo.py` 新增 `resolve_schedule_total_updates()`；新 checkpoint 保存 `max_iterations` 和 `schedule_total_updates`；resume 新 checkpoint 时沿用保存的 schedule horizon；resume 旧 checkpoint 时 fallback 到 `start_iteration`，避免重新打开 entropy。`ecfppo.py` 将 entropy 退火下限 clamp 到 0，避免超过 horizon 后变成负 entropy 系数。
- 验证：`tests/test_train_ecfppo.py` 16/16 通过，新增 `test_resume_schedule_total_updates_preserves_annealed_state()` 覆盖从头训练、新 checkpoint resume、旧 checkpoint resume 和 entropy 下限。
- 后续动作：当前仍在运行的 `20260609-233704` 进程不会自动加载修复后的代码；需要停止后重新从头训练，或用修复后的代码重新 resume。若从头训练，默认 schedule 行为不变。


### D011: bounded actor mean 缓解动作均值越界

- 日期：2026-06-10
- 状态：open
- 严重性：P0
- 触发原因：D010 修复后从头训练 `20260610-080058` 仍在后期出现 actor mean 大面积越界，success 高峰后回落，失败重新由 `no_reach` 主导。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260610-080058/training.log`
- 现象：该 run 完成 500 iter，峰值 `success=0.289`，最终 `success=0.052`；iter 500 `reach_rate=0.060`、`no_reach=0.940`、`act_mean_clip_ratio=0.8408`、`reach_clip_ratio=0.9262`。分组日志显示 `succ/unsafe/noreach` 三组后期均有较高 `mean_clip`，说明越界不是单一失败组独有。
- 初步假设：当前 `Normal(mean, std)` 的 `mean` 是 actor 无界线性输出，而环境在执行前 `torch.clip(action, -1, 1)`；当 `mean` 大面积越界时，PPO log_prob/ratio 仍基于 raw action 分布，真实速度命令和能耗却基于 clipped action，导致策略更新与执行效果错配。
- 代码链路：`EC_EFPPO_ActorCritic.update_distribution()` 生成 `Normal(mean, std)`；`EC_EFPPO.update()` 用同一分布计算 PPO ratio；`HighLevelNavigationEnv.update_velocity_commands()` 和 `update_energy()` 对高层动作裁剪后执行和计能耗；`EC_EFPPO_Buffer.compute_advantages()` 记录 `action_mean_clip_ratio`。
- 证据：外部参考显示 CleanRL PPO 和 rsl_rl 常用无界高斯加环境裁剪，但 Stable-Baselines3/Spinning Up 的 squashed Gaussian 必须同步做 tanh 后的 log_prob 修正。本项目当前只需要先消除 `mean` 越界，直接切完整 squashed Gaussian 会同时改变 log_prob、entropy 和 buffer 语义，风险更高。
- 结论：优先采用 `tanh(mean)` bounded mean 作为最小风险改动。该方案保证 actor mean 落在 `[-1, 1]`，保持采样分布仍为 `Normal(bounded_mean, std)`，不需要改 PPO ratio、buffer 或 checkpoint 参数形状。
- 改动：`rsl_rl/rsl_rl/modules/actor_critic.py` 新增 `bounded_actor_mean`，开启时 `update_distribution()` 和 `act_inference()` 使用 `torch.tanh(raw_mean)`；`legged_gym_go2/legged_gym/envs/go2/go2_config.py` 设置 `bounded_actor_mean=True`；`train_ecfppo.py` 将配置传入 actor-critic；测试新增 bounded mean 覆盖。
- 验证：`tests/test_ecfppo_actor_critic.py` 14/14 通过，`tests/test_train_ecfppo.py` 16/16 通过，`tests/test_ecfppo.py` 20/20 通过。
- 后续动作：用新代码从头跑 300-500 iter。若 `act_mean_clip_ratio` 接近 0 且 `action_clip_ratio` 明显下降，再看 `success/reach_rate/no_reach` 是否改善；若采样动作仍大量贴边，再进入完整 squashed Gaussian 方案，届时必须实现 inverse tanh/log-Jacobian 修正和 entropy 口径更新。


### D012: raw actor mean 饱和导致 bounded policy 仍贴边

- 日期：2026-06-10
- 状态：open
- 严重性：P0
- 触发原因：D011 后训练 `20260610-133801` 显示 `tanh(mean)` 消除了无界越界，但 bounded mean 很快进入饱和边界，策略仍未形成稳定导航。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260610-133801/training.log`
- 现象：截至 iter 446，该 run 峰值 `success=0.115`、`reach_rate=0.196`，后段回到 `success≈0.05`、`reach_rate≈0.065`、`no_reach≈0.935`。`act_clip_ratio` 降到约 0.50，说明 D011 减少了采样动作裁剪；但 debug 从 iter 200 起 `act_mean_clip_ratio≈0.99`，`act_mean_abs≈1.0`，说明 bounded mean 几乎全在 `tanh` 饱和边界。
- 初步假设：D011 的 `tanh(raw_mean)` 只限制了输出范围，没有限制 tanh 前 logits。policy advantage 仍会把 raw actor mean 推到很大，导致 `tanh(raw_mean)` 近似 ±1，动作退化成 bang-bang policy；采样仍有约一半越界被环境 clip，导航策略缺少细粒度控制。
- 代码链路：`EC_EFPPO_ActorCritic.update_distribution()` 计算 raw mean 后 `tanh` 成 bounded mean；`EC_EFPPO.update()` 只对 bounded mean 做旧 `actor_mean_bound_loss`，该 loss 在 bounded mean 下恒为 0；`EC_EFPPO_Buffer` 旧 debug 没有记录 raw mean，无法判断 tanh 前饱和程度。
- 证据：iter 350 是本轮峰值，`success=0.115`、`reach_rate=0.196`，但 debug 显示 `act_mean_clip_ratio=0.9663`；group 中 `succ/unsafe/noreach` 三组 `mean_clip` 均接近 1。iter 440 后 `act_mean_clip_ratio=0.9910`，`noreach=0.938`。
- 结论：不要直接进入完整 squashed Gaussian；如果 raw logits 不受控，完整 squashed Gaussian 也会 tanh 饱和。下一步应先增加 raw mean 诊断和 raw mean 饱和正则。
- 改动：`EC_EFPPO_ActorCritic` 保存 `raw_action_mean`；`EC_EFPPO_Buffer` 保存 raw mean，并新增 `raw_action_mean_abs/raw_action_mean_clip_ratio` 及各维统计；`EC_EFPPO` 新增 `actor_raw_mean_bound` 和 `actor_raw_mean_bound_coef`，在 policy loss 中加入 `relu(abs(raw_mean)-bound)^2`；Go2 默认 `actor_raw_mean_bound=2.0`、`actor_raw_mean_bound_coef=1e-3`；训练日志新增 `raw_mean_bound_loss` 和 raw mean debug 字段。
- 验证：`tests/test_ecfppo_actor_critic.py` 14/14 通过，`tests/test_train_ecfppo.py` 16/16 通过，`tests/test_ecfppo.py` 21/21 通过。
- 后续动作：从头跑 200-300 iter。若 `raw_mean_clip_ratio` 降低且 `reach_rate` 恢复，继续 500 iter；若 raw logits 仍饱和，优先提高 raw 正则或降低 policy LR/std，而不是继续长训。

## 决策记录

- 2026-06-05：确认 `std` 未加入 policy optimizer 是确定实现 bug，已按最小修复处理。该改动不改变 EC-EFPPO 的 GAE/target 语义，只恢复参考实现中 policy 分布参数可训练的基本行为。
- 2026-06-05：新增诊断日志属于观测性改动，用于后续判断 reach critic 发散、energy 饱和和 `done_for_gae` 过密是否仍存在。
- 2026-06-05：临时训练已证明 `std` 修复生效，但没有证明训练稳定性已解决；下一轮应优先分析 energy 饱和和 done mask。
- 2026-06-06：确认直接优化实际 `std` 会造成探索噪声无界增长。EC-EFPPO 改为优化 `log_std`，通过 `exp(clamp(log_std))` 得到标准差，并降低/退火 entropy，先消除策略分布层面的不稳定来源。
- 2026-06-06：最新训练证明 `std` 已受控，但 reach target/bootstrap 发散与 energy/done 饱和仍导致训练坍塌。下一步只增加分组诊断统计，不改变训练语义。
- 2026-06-06：已增加 D003 分组诊断统计并通过单元测试；50 iter 小规模短训完成。短训未复现 reach loss 爆炸，但确认 energy/done 饱和稳定存在，且新增分组字段可用于后续大规模日志判定。
- 2026-06-07：20260606-230145 大批量日志证明极端 reach target 来自极少量 open/bootstrap 样本。已增加 `reach_value_clip=5000.0` 作为 bootstrap value 语义边界，并记录 `reach_clip_ratio` 用于下一轮诊断。
- 2026-06-07：20260607-094159 完整训练证明 `reach_value_clip` 抑制了 target 爆炸，但 success 仍坍塌；下一步进入 D004，只增加 energy/action 诊断字段，不继续长训、不继续调 `reach_value_clip`。
- 2026-06-07：已完成 D004 100 iter 诊断短训。完整统计显示平均每步耗能约 116、最大恒为 120、平均第 5.67 个高层步触到能量下界、动作裁剪比例约 0.953，能耗尺度过大假设被直接验证。
- 2026-06-07：D004 第一轮配置实验：将 Go2 高层 `energy_consumption_scale` 设为 `8.0 / (3 * high_level_action_repeat) ≈ 0.5333`，目标是把单个高层步最大耗能从 120 降到约 8。
- 2026-06-07：20260607-225408 短训确认能耗缩放生效，首次触底步数从约 5.67 推迟到约 77.99，open 样本比例从约 0.00484 提升到约 0.078；但动作裁剪比例仍约 0.954，下一步优先降低探索噪声/动作饱和。
- 2026-06-08：进入 D005。先把 `log_std_max` 从 0.0 降到 `log(0.5)`，同时新增动作均值和裁剪后动作统计，用于区分动作贴边来自探索噪声还是 actor mean 越界。
- 2026-06-08：20260608-095413 短训证明 `std` 已受控但动作贴边未改善，且 `act_mean_clip_ratio≈act_clip_ratio`；下一步优先降低 policy learning rate 或约束 actor mean。
- 2026-06-08：D005 第二轮改动：拆分三路 optimizer 学习率，将 policy learning rate 从 1e-3 降到 3e-4，energy/reach learning rate 暂保留 1e-3，用于验证 actor mean 跑飞是否来自 policy 更新过强。
- 2026-06-08：20260608-113056 短训证明 policy 降速有效抑制 actor mean 极端爆炸，但动作均值越界比例仍约 0.897；下一步建议继续把 policy learning rate 降到 1e-4，先于 actor mean 结构约束。
- 2026-06-08：D005 第四轮改动：将 policy learning rate 从 3e-4 继续降到 1e-4，其他关键配置保持不变；下一轮只跑 100-150 iter 短训验证动作贴边是否继续缓解。
- 2026-06-08：20260608-133653 短训证明 `policy_learning_rate=1e-4` 明显改善动作贴边和能耗，但 success 更早坍塌，`reach_clip_ratio` 与 `reach_loss` 恶化；停止继续单纯降低 policy 学习率。
- 2026-06-08：新增 D006，下一步优先降低 reach critic 更新强度，例如先将 `reach_learning_rate` 从 1e-3 降到 3e-4，再跑 100-150 iter 短训。
- 2026-06-08：D006 第一轮改动：将 `reach_learning_rate` 从 1e-3 降到 3e-4，保持 policy/energy 学习率和其他关键配置不变，用于验证 reach critic 输出越界是否来自更新强度过大。
- 2026-06-08：20260608-151933 短训证明降低 reach learning rate 有效，`reach_loss` 和 `reach_clip_ratio` 明显下降，success 后段改善；下一步先做 300-500 iter 中等规模复测，不立即继续改结构。
- 2026-06-09：20260608-221634 续训到 total iter 500 后，success 最终约 0.052，`act_mean_clip_ratio` 后期约 0.97，`energy_loss` 和 `reach_loss` 均重新升高；停止继续长训，下一步优先 actor mean 边界正则，而不是继续单独调 reach critic。
- 2026-06-09：D007 第一轮改动：在 policy loss 中加入 actor mean 边界正则 `mean_bound_loss`，默认 `actor_mean_bound=1.0`、`actor_mean_bound_coef=1e-3`；该改动不改变环境裁剪和高斯策略分布定义，只惩罚越界均值。
- 2026-06-09：20260609-080519 验证 `actor_mean_bound_coef=1e-3` 方向有效但偏弱；动作均值越界和能耗显著低于无正则续训，但后期仍回升并导致 success 坍塌。
- 2026-06-09：D007 第二轮改动：将 `actor_mean_bound_coef` 从 `1e-3` 提高到 `1e-2`，保持 policy/reach/energy 学习率、std 上界和环境动作裁剪逻辑不变，用于验证更强边界正则能否长期压住 actor mean 越界。
- 2026-06-09：20260609-121135 证明 `actor_mean_bound_coef=1e-2` 显著稳定动作、能耗和 reach critic，但 success 在 151-160 高峰后仍断崖坍塌；下一步不继续盲目加正则，先补 success 分解和动作分量诊断。
- 2026-06-09：D007 第三轮改动：已补 success 分解诊断和动作分量诊断，且不改变训练逻辑；下一轮用 100-150 iter 短训判断 success 断崖主要来自未到达、不安全，还是某一动作维度异常。
- 2026-06-09：运行中短训 `20260609-144319` 早期显示 `no_reach` 很高且第 0 维 `act_mean_clip_dim` 明显异常；该信息暂作为下一步分析方向，必须等待 100-150 iter 完成后再形成最终结论。
- 2026-06-09：D008 确认 EC-EFPPO policy loss 使用的 `advantages_total` 符号方向与 reach-avoid cost-like 语义不一致；已将标准化后的 `advantages_total` 取负再送入 PPO policy loss，并用小张量梯度测试锁定方向。
- 2026-06-09：`20260609-174808` 证明 D008 后目标驱动恢复，`reach_rate` 峰值到 0.688；当前瓶颈转为未到达和 `unsafe_before_reach` 并存，进入 D009。
- 2026-06-09：D009 只新增 `succ/unsafe/noreach` 分组诊断日志，不改变训练语义；冒烟训练 `20260609-213244` 已确认 `group` 行正常写入。
- 2026-06-10：D010 修复 resume 续训 schedule 语义；新 checkpoint 保存 `schedule_total_updates`，旧 checkpoint resume 不再按新的 `max_iterations` 重新打开 entropy。
- 2026-06-10：`20260610-080058` 干净从头训练仍显示后期 actor mean 大面积越界；D011 先采用 `tanh(mean)` bounded mean，不直接上完整 squashed Gaussian，避免同步改 PPO log_prob、entropy 和 buffer 语义。
- 2026-06-10：`20260610-133801` 证明 D011 降低了采样动作裁剪但 bounded mean 迅速 tanh 饱和；D012 增加 raw mean 诊断和 `actor_raw_mean_bound` 正则。

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
