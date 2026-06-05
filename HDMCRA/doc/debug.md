# HDMCRA Debug Log

本文档用于记录 HDMCRA 当前阶段的训练稳定性诊断。旧的实现计划和历史 debug 记录已经不再作为当前判断依据；后续分析从本文档重新开始。

## 当前状态

- EC-EFPPO 的主体实现已经完成，可以运行端到端 Go2 训练。
- 当前尚未证明训练可以稳定达到 Reach-Avoid PPO 基线水平。
- 后续工作重点是基于训练日志和代码链路，判断算法语义、环境信号、超参数、网络结构或统计口径中是否存在不合理设计。
- 任何改动都需要先形成可验证假设，再通过测试或训练结果验证。

## 当前待分析问题

| ID | 状态 | 严重性 | 问题 | 相关日志/文件 | 下一步 |
|---|---|---|---|---|---|
| D001 | open | P0 | 最新训练中 success 偏低且 value loss 量级异常 | `legged_gym_go2/logs/ecfppo_go2/20260605-102937/training.log` | 分析 `reach_loss`、`energy_loss`、value target 和 advantage 量级 |

## 训练记录索引

| Run ID | 日期 | 命令/配置 | 迭代范围 | peak success | final success | 关键异常 | 结论 |
|---|---|---|---|---|---|---|---|
| 20260605-102937 | 2026-06-05 | `train_ecfppo.py --headless --num_envs 4096 --max_iterations 1500` | 待分析 | 待分析 | 待分析 | `energy_loss` 和 `reach_loss` 量级异常 | 待分析 |

## 分析记录

### D001: 最新 EC-EFPPO 全量训练稳定性分析

- 日期：2026-06-05
- 状态：open
- 严重性：P0
- 触发原因：最新训练日志显示 success 未接近基线水平，同时 value loss 量级异常。
- 相关日志：`legged_gym_go2/logs/ecfppo_go2/20260605-102937/training.log`
- 现象：待补充完整日志统计。
- 初步假设：待补充。
- 代码链路：待补充。
- 证据：待补充。
- 结论：待补充。
- 改动：暂无。
- 验证：暂无。
- 后续动作：先统计训练日志，并检查 `targets_reach`、`targets_energy`、`advantages_total`、`g_values`、`h_values` 的量级。

## 决策记录

当前暂无新的设计决策。后续如果修改算法语义、环境信号、超参数、网络结构或统计口径，需要在这里记录原因和结果。

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
```

<!--
示例记录，后续正式记录时可以参考这个结构，不要把本注释当作真实结论。

### D002: reach critic target 量级异常

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
-->
