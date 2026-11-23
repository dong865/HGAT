# V2_SCHEMA — continuous_membership_v2 契约文档

架构标识：`continuous_membership_v2`
本文档对应交付物要求第 3 项。所有张量按全局任务 ID 组织。

## 1. ID 契约

| 对象 | 表示 |
|---|---|
| 未分配 | `assignment[i] == -1` |
| 真实任务 | 全局任务 ID ∈ `[0, max_tasks)`，代码中 0 是合法任务，**不是**空目的地 |
| 空目的地 | padded `full_plan` / `hard_full_plan` 的**最后一列**（索引 `max_tasks`） |
| 压缩求解 | Sinkhorn 与硬指派在活跃子集上求解，返回前按 `active_agent_ids` / `active_task_ids` 显式还原 |

## 2. 核心张量

| 张量 | 形状 | 说明 |
|---|---|---|
| `demand_raw` | `[Nt_max]` | 需求头 softplus 输出 × 任务 mask，非活跃 = 0 |
| `capacity_task` | `[Nt_max]` | `alpha·d_j`，非活跃 = 0；`alpha=min(1,N/Σd)` |
| `capacity_idle` | 标量 | `N − Σ b_j`，clamp_min(0) |
| `task_plan` (P) | `[Np_max, Nt_max]` | 软计划，仅真实任务列 |
| `full_plan` (P̄) | `[Np_max, Nt_max+1]` | 末列 = 空目的地 |
| `row_residual` / `col_residual` | `[Np_max]` / `[Nt_max+1]` | 边际残差，>1e-3 抛错，不静默掩盖 |
| `hard_full_plan` (X̄) | `[Np_max, Nt_max+1]` | 二进制，一人一归属 |
| `assignment` | `[Np_max]` int | 全局任务 ID 或 -1 |

## 3. 需求语义

- `d_j`：状态相关的**期望资源量/分配偏好**（非负连续标量），不预设最低可行人数档位；
- 共同尺度在供给饱和分支不可辨识——论文措辞用"需求偏好/期望需求"，不作"严格最低人数"主张；
- 初始化：末层 bias = softplus⁻¹(demand_init=3.0)，工程初值，6v2 主场景起步值。

## 4. 观测 schema（V2）

块结构：`[raw_available_features, membership_conditions, h_i, t_i, c_i, retained_geometric_guidance]`

- 删除 TRK/BLK/CAP 阶段 one-hot（5 维）与 Core/Support/Reserve 角色 one-hot（3 维）；
- `membership_conditions`：当前分配有效标志、实际同任务人数（按可用团队规模归一化，不再固定除以 3）；
- `(t_i, c_i)` 由**同一聚合函数** `build_membership_context(K=P 或 X)` 产生，未分配成员两块为零，自身几何与图表示保留；
- 不直接把 `demand_raw` / `capacity_task` / 关系代价拼进 Actor/Critic 输入；
- 保留环形目标点几何特征（固定引导，记录于配置）；
- 版本号：`observation_schema_version = v2_no_stage_role`（新训练重新计数 obs 维度）。

## 5. Checkpoint 元数据（新增）

```text
architecture_version        = continuous_membership_v2
observation_schema_version  = v2_no_stage_role
task_plan_schema_version    = continuous_ot_v1
membership_schema_version   = convex_count_assignment_v1
capture_definition_version  = assigned_members_geometric_v1
context_pool                = normalized_co_membership_mean
target_context              = hard
demand_parameterization     = scalar_softplus
```

标准 resume 检查以上字段；旧三分类 demand_head 与新标量头不兼容，禁止 `strict=False` 静默迁移。

## 6. 求解与门控参数（第一版）

| 参数 | 默认 | 说明 |
|---|---|---|
| epsilon_OT | 0.5 | Sinkhorn 温度（与训练一致，对照实验再调） |
| sinkhorn_iters | 80 | 残差 >1e-3 报错 |
| count_penalty λ_n | 1.0 | 凸人数偏差代价（量纲与旧 soft-score 不同，切换时需重校） |
| switch_penalty λ_sw | 0.5 | 自愿切换惩罚；失活/离线导致的被迫释放单列统计，不受其阻挡 |
| hysteresis | 0.05 | 能量改善阈值 |
| min_dwell_steps | 5 | 最小驻留 |
| assign_interval | 5 | 可配置，不再写死 `% 5` |
| plan_eps | 1e-8 | log 计划地板（仅硬评分） |
