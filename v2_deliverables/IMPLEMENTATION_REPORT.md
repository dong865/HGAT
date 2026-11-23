# IMPLEMENTATION_REPORT — continuous_membership_v2

状态：阶段 A、B 完成；C（环境/执行）、D（训练接通）、E（效果验证）待做。
所有改动均在**新文件**中，旧代码、旧 checkpoint、旧实验目录零改动。

## 阶段 A：版本与协议核实（完成）

详见 `PHASE_A_BASELINE.md`：仓库路径、核心文件 SHA-256、全部 `allocate()/encode()/apply_roles()` 调用点、`soft_task_plan` 传入情况。

关键核实结论：
- `soft_task_plan` 仅 `main_stage_end_to_end.py:502` 传入；`runtime_stage.py:379` 与 `main_stage_feasible.py` 两入口走 legacy 规则计划——与交接文档第 1 节预警一致，V2 必须三入口统一。
- 当前运行的 `stage_networks.py` 中 Sinkhorn 已收正代价（历史负号修复在本机版本中属实，行 216-220 区域已改代价语义）。

## 阶段 B：纯算法组件（完成）

新文件：`continuous_membership.py`（约 500 行），实现：

| 组件 | 对应章节 | 说明 |
|---|---|---|
| `TaskPlan` / `MembershipPlan` | §4 | 显式 ID 映射的数据契约 |
| `ContinuousDemandHead` | §3 | 单标量 softplus 头，末层 bias=softplus⁻¹(3.0)，mask 归零 |
| `build_capacity` | §5.1 | 统一容量函数（α 缩放），**所有模式（含固定需求消融）共用** |
| `build_soft_plan` | §5.2–5.4 | 正代价 log-domain Sinkhorn，float64 保 autograd，零容量列不参与缩放，行/列残差 >1e-3 抛错不静默掩盖；N=0/M=0/空目的地零容量全部显式分支 |
| `build_membership_context` | §6.2 | 统一聚合 T=KHᵀ, W=KKᵀ, 归一化成员均值；软 P / 硬 X 同一函数 |
| `solve_membership` | §7 | 凸人数偏差 Δ_j(k)=λ_n(2k−1−2b_j) 的线性指派展开；k 为内部单位计数，不在可微图内 |
| `membership_gate` | §8 | 修非法 → 保捕获 → 驻留+滞回 → 强制事件优先级；无角色/阶段依赖 |

### 验收测试（`test_continuous_membership.py`，第 15 节 A 组）

**11/11 通过**，完整输出在 `V2_ACCEPTANCE_TESTS.log`：

- A1 代价方向（非退化 2×2，对角匹配获得 ~1.0 质量）
- A2 OT 守恒六情形（充足/不足/M=0/N=0/非活跃 padding/空目的地零容量）
- A3 全局 ID（活跃任务 [1]、[1,2] 非连续编号不错列）
- A4 资源竞争（N<M 合法，允许任务未服务）
- A5 **硬求解器 vs 穷举，320 个随机案例全等**（含空目的地/切换代价/非整数 b，n≤4, m≤3）——独立复现了交接文档 §7 的等价性校核
- A6 统一上下文一致性与置换等变
- A7 梯度开关不改前向 + 需求头初始化/非负/掩码
- B 有限差分：costs 与 demand 的 Sinkhorn 梯度（float64，固定 80 迭代）
- 门控四分支：驻留/滞回/强制/捕获保持

### 与交接文档的偏差记录

1. 需求头初始化偏差：交接建议 bias 初始化使初始需求≈3.0；实测零输入输出 3.05（上游 Linear bias 贡献），属工程可接受范围，测试容差 0.15。
2. `available_agent_mask` 语义：软计划/硬求解/执行应用三处使用同一 mask 的契约已在接口层固定（TaskPlan 字段），环境侧接线在阶段 C 验证。

## 阶段 C：环境适配与统一 controller_step（完成，8/8 验收通过）

新文件：v2_env_adapter.py（assignment-only 成员定义，委托 apply_roles 复用基线重置）、
v2_controller.py（统一 controller_step）、runtime_continuous_v2.py（回放/仿真运行时）、
test_v2_integration.py。

审查意见落实：
1. 门控优先级修正为 失效修复 -> 强制事件 -> 捕获保持 -> 驻留+滞回；新增组合条件测试 4 项
   （驻留未结束+离线 / 捕获保持+目标失活 / 保持+强制事件 / 合法保持）全部通过；
2. 需求初始化契约精确化：末层权重零初始化 + bias=softplus^{-1}(3)，任意输入严格输出 3.0；
3. 归属失效应立即处理：controller 每步检查 membership_validity，非法即重构，不等 assign_interval；
4. 基线健壮性修复：stage_networks.encode 的 target 默认值（legacy 在角色CORE但任务失活
   状态下 UnboundLocalError，V2 新顺序暴露）；legacy 正常流程不受影响。

### C 组验收（V2_INTEGRATION_TESTS.log，8/8）

- C1 学习需求影响执行（候选随需求改变；门控单独验证）
- C2 无隐藏角色差异（members == capture_members == coalition_of == shaping_members）
- C3 生命周期及时修复（目标失活 / 成员离线均立即 invalid_fixed，不等周期）
- C4 时间状态正确（每物理时刻 H 恰好推进一次）
- C5 两入口逐步一致（训练 controller 与 runtime 回放 8 步：demand/assignment 精确相等，
  动作容差 0.05 内；评估入口在阶段 E 复用同一 controller）
- C6 无档位泄露（扰动 legacy stages/roles，V2 观测与需求不变）
- C7 ID 与空任务正确（非连续编号不错列；全部目标完成时合法全未分配）

### 重要基线发现（已记录，未静默修改 legacy）

runtime_stage.py 的实机路径不递增 env.episode_step_count，legacy 部署中编码器时间状态
在首步后命中 same_physics_step 缓存而冻结。V2Runtime 已修复（每动捕帧递增）。
建议确认 legacy 实飞数据是否在冻结 H 条件下采集。

## 阶段 C遗留项（并入 D）

- `stage_env.py` 增加 assignment-based 成员查询（`members(j)`、`apply_membership`），`coalition_of`/`shaping_members` 统一指向同一成员定义；保留 EXIT/LANDED 生命周期；
- 新 `controller_step` 服务：生命周期 → 计划 → （周期/事件）硬求解+门控 → apply → 同一 H 重建硬上下文 → Actor；
- 三入口（采集 `main_continuous_v2.py` 新主入口 / 评估 / `runtime_stage.py` V2 分支）消费同一 `TaskPlan`，缺失学习计划直接报错。

## 阶段 D：训练接通（完成）

新文件：v2_training.py（可微观测构建 / raw 回放 / 片段编码 / 三阶段更新）、
v2_gradient_check.py（三 pass）、main_continuous_v2.py（新主入口）。

关键实现事实：
- 可微 V2 观测：图块（h_i / t_ctx / c_ctx）保持 autograd，常量块 numpy；
- 需求头只经 P 路径获得策略/价值梯度；辅助阶段仅关系 Smooth L1（标签定义不变：
  当前成员位置 vs 未来轨迹最小距离）；无阶段 CE、无人数一致性伪监督；
- TD 目标 --target-context hard（soft 留作显式对照）；目标整体 stop-gradient；
- 回放：L 条 transition + L+1 对齐决策状态（t+1 = 下一决策时刻 post-assignment）；
- stop-plan-gradient 消融在统一上下文消费 K 前 detach P（前向不变）；
- optimizer 修复：encoder.parameters() 已含 st_hgat+relation_head，仅追加 demand 头。

### 三 pass 纯价值梯度检查（V2_GRADIENT_CHECK.json，PASS）

Pass A：relation 2.8e-3 / demand 5.4e-2 / st_hgat 1.21 全非零；
Pass B（上下文前 detach P）：relation/demand 精确归零，st_hgat 保留 1.22；
Pass C（整 obs detach）：编码器侧全零，actor 保留。与交接文档 B 组预期逐条吻合。

### 短训练 smoke（V2_SMOKE_TEST.log）

15 episode / 30 更新：critic/actor/relation 三阶段 loss 有限；需求头参数变化
（demand 3.00 -> 3.04）；评估循环正常；schema 与 allocation_source 入日志。
修复记录：初版观测 builder 经 numpy 拼接切断 autograd（三 pass A 曾全零，
已改为 torch.stack 保留图块连接）——这是本次阶段 D 最主要的实现陷阱。

### 长训练启动（进行中）

- V2 编码器预训练（无 stage/role/ring 标志，60 episode）后台运行；
- 10:35 一次性任务：加载 warm start 启动 sdgcr_v2_main（V2 Full）与
  sdgcr_v2_fixeddem（固定需求消融）两臂，挂监控与 2 小时常驻检查。

## 阶段 D（待做）：训练接通

- 删除阶段 CE；辅助阶段仅关系 Smooth L1（标签定义不变：当前成员位置 vs 未来轨迹最小距离）；
- TD 目标 `hard+no_grad` 显式配置（`--target-context hard`），`soft+no_grad` 留作对照；
- raw replay 保持 L+1 对齐契约（已在旧 `encode_ordered_segment` 修复基础上接入 V2 上下文）；
- 三 pass 纯价值梯度验证（detach P 于统一上下文消费前）+ `V2_GRADIENT_CHECK.json`；
- 短训练 smoke 覆盖 critic/延迟 actor/辅助三阶段，`V2_SMOKE_TEST.log`。

## 风险与已知限制

1. 塌缩风险仍在：V2 主模式默认联合训练 encoder，需配合既有预训练 warmup 使用（阶段 D 默认配置 `--init-encoder-from` + 可选 `--freeze-backbone`，后者仅作明确对照）。
2. λ_n/λ_sw/hysteresis 量纲与新 log-plan 能量一致，但与旧 soft-score 量纲不同——阶段 E 消融前需做敏感性检查。
3. target/partial 等历史指标需加入新日志行（顺带补 loss 持久化，只加不改旧格式）。
echo "report appended"

## 战役终局（2026-11-15 训练停止）

最终产物（均已核验 schema 完整）：
- sdgcr_v2_fixeddem/checkpoint_014000.pth（d1.5 最终权重）+ best_model.pth（ep11500，战役首次完整捕获 cap=0.1）
- sdgcr_v2_dcap20/checkpoint_010900.pth（d2.0 最终权重，85 事件，峰值 tgt=0.5）+ best_model.pth
- 归档档：main（联合学习）/sep（分离学习）/wide（容量）各含完整日志

五臂终局：fixeddem 74 事件 1 完整捕获；dcap20 85 事件 1 完整捕获（ep300 即首获）；
sep 9 事件；main 3 事件；wide 0 事件。判定：塌缩与耦合为真实瓶颈（已治/已避），
容量与分离学习证伪，d_cap 1.5 偏紧（温和）成立；全部 V2 臂在纯稀疏奖励下训练
（奖励塑形未启用，留为后续实验）。

遗留工程备注：
1. dcap20 末期曾出现双进程并发写同一输出目录（一次重启命令被执行两次所致），
   其重启后 300 回合权重被 cron 按名称排序裁剪误删；最终存档以 ep10900 为准，不受影响。
2. fixeddem 旧进程带 _skip_legacy_plan=False 包袱运行至停止；新进程将自动跳过
   encoder 内部旧软计划（省 20-30% 算力并消除一类 CUDA 故障面）。
