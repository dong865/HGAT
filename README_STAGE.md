# SDGCR：核心-支援-预备分层的阶段可行动态图联盟重构（论文主线方法）

本文方法是修订后 HGAT 论文（*main_core_support.tex*）的代码实现，与旧路线
（固定 6V2 + 严格 3+3 + 组间参数交换 MADDPG）不同，核心是：

```
固定规模分布（P(6v2)=0.7, P(4v2)=P(5v2)=P(6v3)=0.1，无课程）
  → 核心-支援-预备角色分配（κ(m)只约束核心；掩码Sinkhorn [C^c|C^s|C^r]；
     离散投影+滞回/驻留；目标切换与角色切换分别计量）
  → 角色条件化共享Actor + 集合不变Critic（obs含角色嵌入，预备队用风险加权集合）
  → 私有目标逃逸者（任一到达目标即追捕失败）
  → 任务图/安全图双生命周期（捕获→释放→退出→降落；任何追捕者不因释放停止运动）
  → 50 Hz 速度指令（cmdVelocityWorld 语义）
  → 3-D 状态（z 轴 + 定高飞行控制器，动作保持 2-D [vx, vy]）
```

与论文公式的对应关系：

| 论文公式 | 代码位置 |
|---|---|
| 核心容量 κ(m)：trk=1, blk=2, cap=3, exit/landed=0 | `stage_env.mode_capacity` |
| 最大逃逸开口 Ψ_j | `stage_env.escape_gap` |
| 核心/支援/预备角色互斥穷尽 Σ(X^c+X^s)+R=μ | `stage_coalitions.StageCoalitionAllocator` |
| 核心可行性 F_trk, F_blk, F_cap | `stage_coalitions.StageCoalitionAllocator.evaluate_feasibility` |
| 核心边代价 C^c（截获+障碍+不确定+切换+图关系） | `stage_coalitions.StageCoalitionAllocator.edge_costs` |
| 支援槽位 q^sup（预测逃逸方向两侧） | `stage_coalitions.support_slots` |
| 支援边代价 C^s、机动预备代价 C^r（风险加权到达时间，非静止） | `stage_coalitions.StageCoalitionAllocator.edge_costs` |
| 角色掩码Sinkhorn [C^c\|C^s\|C^r]（预备列=角色非idle） | `stage_coalitions.MaskedSinkhorn` |
| 离散投影 max<P*,Z>−λ_tar N_tar−λ_role N_role + 滞回/驻留 | `stage_coalitions.StageCoalitionAllocator._project/_score` |
| 角色条件化Actor a_i=π(h_i,h_j,h_C,e(m),e(r)) | `stage_networks.SharedActor`（obs含role one-hot） |
| 集合不变Critic Q=ρ(Σ μ ψ(h_i,a_i,e(r),e(m))) | `stage_networks.SetInvariantCritic` |
| 归一化团队奖励 r_t | `stage_env.step`（team_reward） |
| 私有目标 + 逃逸失败终止 | `scenario_protocol.goal_reached` + `stage_env.step` |
| 任务图/安全图双掩码 + 释放转机动预备 | `stage_env.advance_lifecycle` |
| 切换连续性 + 安全投影 | `runtime_stage.StageCoalitionRuntime._rate_limit` + `safety.filter_velocity` |

## 训练

```bash
cd "/media/yudong/data/ros2_ws_6V2"
export PYTHONPATH="$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH"
PYTHON=/home/yudong/anaconda3/envs/uav/bin/python

$PYTHON -m crazyflie_examples.six_v_two_hgat.main_stage_feasible \
  --episodes 20000 --max-steps 1000 --batch-size 256 \
  --lr-critic 3e-4 --critic-reward-scale 0.1 --critic-reward-clip 10.0 \
  --policy-delay 2 \
  --actor-action-norm-weight 0.1 \\
  --device auto --output checkpoints/stage_feasible --seed 0
```

- 全程不使用课程学习；每回合从固定混合分布采样：6追2占70%，4追2、5追2、
  6追3各占10%（论文实验协议）。
- train/validation/test 种子空间隔离（`derive_scenario_seed`），同一场景种子
  同时确定障碍物、出生点与私有目标点。
- 每个逃逸者有私有目标；任一活动逃逸者到达目标立即终止整局并记为追捕失败。
- TensorBoard 默认写入 `runs/`；`--no-tensorboard` 可关闭。
- 环境原始奖励用于论文统计；critic 回放奖励按 0.1 缩放并裁剪到
  `[-10,10]`，actor/目标网络每两次 critic 更新一次，以抑制 Q 值与策略漂移。
- TensorBoard 记录 Q/TD、梯度裁剪、动作饱和率和逐项奖励。

## 3-D 状态说明（v6 检查点）

自 v4 起 `StageCoalitionEnv` 的状态是 **3-D**（`positions`/`velocities` 为
`(N, 3)`，z 轴默认巡航高度 `flight_height=1.0` m），但**动作保持 2-D
`[vx, vy]`**（与论文 planar 几何一致）：

- z 轴由环境内置的弹簧-阻尼定高控制器驱动，并叠加小幅随机扰动，模拟真实
  飞控导致的高度波动（`height_kp`/`height_kd`/`height_noise`）；
- 捕获判定、角度开口 Ψ_j、支援槽位、IBR 规划等**水平面几何**仍按 xy 计算，
  距离（截获、d_cap、碰撞）按 3-D 计算；
- 观测新增 z/vz 特征：self 8→10、teammate 2→3 每机、target 2→3，`obs_dim`
  151 → **159**；图节点特征保持 planar（`node_dim=6`）；
- `real_flight.py` 把完整 3-D mocap 状态传给 `runtime_stage`，z 由真实飞控
  控制，规划层只发平面速度；
- 检查点版本 v3 → **v4**，旧 v3 检查点不兼容（`runtime_stage` 仍可读 v2/v3
  旧模型，但新训练产出 v4）。

### 性能提示（CPU）

本机 torch 在 CPU 上对小型网络存在**线程过载**问题（默认 24 线程下单个
LayerNorm 约 47 ms）。所有训练/评估入口已加入 `torch.set_num_threads(4)`，
单步从 ~500 ms 降到 ~3 ms。GPU 训练（`--device auto`）不受影响。

## 对比基线与消融（公平协议）

所有基线（含 SDGCR）共用同一个 `StageCoalitionEnv`（同一 8×8 场地、10 障碍、
50 Hz、1 000 步、捕获判定 d_cap=2.0 m / 保持 5 步（0.1 s）、私有目标逃逸失败、同一
归一化奖励与终止结果），保证对比公平。基线均固定 6v2 训练；SDGCR 用混合分布
训练，评估时在 6v2 列与基线对比（其余规模作为泛化展示）。

### 5 个独立主对比基线

正式外部方法为 IBR、MAPPO、标准 MADDPG、G2ANet Actor-Critic 和
CBBA+IBR。当前本轮已验证的入口是 IBR 与 MAPPO；其余三个在注册表中明确标记
为 `implemented=False`，完成忠实复现前统一评估器会拒绝运行，避免误用旧结构
变体冒充外部基线。Grouped/Independent/No-Graph/Role-Based/Unified-Group
仅保留旧检查点兼容，不再计入论文主基线数量。

```bash
# 1) IBR 博弈求解器（无学习，无需训练）
$PYTHON -m crazyflie_examples.six_v_two_hgat.main_ibr_baseline \
  --episodes 500 --seed 17 --scales 6v2,4v2,5v2,6v3

# 2) MAPPO（共享Actor + 真集中式Critic）
$PYTHON -m crazyflie_examples.six_v_two_hgat.main_mappo_baseline \
  --episodes 20000 --max-steps 1000 --seed 0 --device auto \
  --output checkpoints/mappo_baseline

# 3)-5) maddpg / g2anet_ac / cbba_ibr
# 将在独立的基线忠实复现计划完成后开放正式训练和评估入口。
```

### 5 项正式消融（一次只改变一个模块）

```bash
for ABL in no_graph mean_pool fixed_concat fixed_coalition no_feasibility; do
  $PYTHON -m crazyflie_examples.six_v_two_hgat.main_stage_feasible \
    --episodes 20000 --max-steps 1000 --seed 0 --device auto \
    --ablation "$ABL" --output "checkpoints/abl_${ABL}"
done
```

规范名称依次对应：A1无图消息传递、A2均值池化、A3固定身份槽位聚合、
A4固定联盟成员、A5无可行性投影。A1–A5与完整方法共用同一训练预算、
种子、奖励和捕获协议。旧角色开关仅用于补充实验，通过
`--supplementary-ablation`调用，不得与正式A1–A5混合作为主表结果。

### GPU 加速说明

`--device auto` 会把下列运算放到 GPU：ST-HGAT/预测器/Actor/Critic 前向、
训练 batch、分配器掩码 Sinkhorn、IBR 候选批量评估。环境动力学（numpy，
小规模）与离散投影枚举（Python）保留在 CPU；`cmdVelocityWorld` 实机推理仍
以 CPU 运行并校验 p95 时延。

## 评估

```bash
$PYTHON -m crazyflie_examples.six_v_two_hgat.evaluate_stage_feasible \
  checkpoints/stage_feasible/best_model.pth --episodes 10 --seed 17
```

评估会分别运行6追2、4追2、5追2和6追3；`--episodes`表示每个规模的回合数。最佳模型评分为`0.7×R_6v2 + 0.3×四规模宏平均`。

## 冒烟测试

```bash
$PYTHON -m crazyflie_examples.six_v_two_hgat.smoke_test_stage
```

覆盖：变规模环境、生命周期、掩码Sinkhorn守恒、分配滞回、网络变规模前向、
短训练循环、检查点往返、实机运行时接口。

## 实机部署（8 架 Crazyflie：6 追 + 2 逃）

```bash
$PYTHON -m crazyflie_examples.six_v_two_hgat.real_flight \
  checkpoints/stage_feasible/best_model.pth \
  --method stage_feasible_dgcr \
  --roles cf1,cf2,cf3,cf4,cf5,cf6,cf7,cf8 \
  --obstacles-file src/crazyswarm2/crazyflie_examples/config/virtual_obstacles_6v2.json \
  --duration 20 --height 1.0 --device cpu
```

- 前 6 个名字为追捕者，后 2 个为逃逸者；逃逸者默认势场逃逸
  （`--evader-mode stationary` 可改为静止目标）。
- 目标捕获后任务掩码立即置 0（联盟可重构），安全掩码保持到该机飞往场边
  降落区为止（双图生命周期）。
- 旧路线检查点仍可用：`--method legacy`。

## 性能说明

- 分配重构默认每 5 个控制周期执行一次（论文的低频重构设计），训练/实机一致。
- CPU 上单步约 30–65 ms（ST-HGAT 注意力为瓶颈）；建议训练用 GPU
  （`--device auto` 自动选择），实机推理用 CPU 并确认 p95 时延低于 20 ms。
- 规模上限：Np≤6、Ne≤3（9 个实体 + 10 个障碍物节点）。
# Encoder pretraining and end-to-end gradient probe

The graph representation can be pretrained without behavioural cloning.  The
supervision is the delayed future-interception relation already defined by the
SDGCR encoder; exploratory actions only provide state coverage.

```bash
cd /media/yudong/data/ros2_ws_6V2
export PYTHONPATH="$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH"
PY=/home/yudong/anaconda3/envs/uav/bin/python

$PY -m crazyflie_examples.six_v_two_hgat.pretrain_stage_encoder \
  --episodes 200 --max-steps 200 --update-interval 25 \
  --device auto \
  --output checkpoints/encoder_pretrain_6v2 \
  --run-name SDGCR_encoder_pretrain_6v2
```

Use the trained representation with ordinary encoded replay by loading and
freezing it.  `--ring-slot-interface` is required because the pretraining
checkpoint was built with the 162-dimensional ring-slot observation layout.

```bash
$PY -m crazyflie_examples.six_v_two_hgat.main_stage_feasible \
  --episodes 2000 --max-steps 1000 --device auto \
  --scale-weights 1,0,0,0 --validation-scales six_v_two \
  --ring-slot-interface --lock-on-close3 \
  --init-encoder-from checkpoints/encoder_pretrain_6v2/best_encoder.pth \
  --output checkpoints/encoder_frozen_6v2 \
  --run-name SDGCR_encoder_frozen_6v2
```

The separate experimental entry point re-encodes ordered raw trajectory
segments with current weights.  It logs `grad_norm/*` and
`parameter_delta/*` for ST-HGAT, relation head, Actor, and both Critics.

```bash
$PY -m crazyflie_examples.six_v_two_hgat.main_stage_end_to_end \
  --init-encoder-from checkpoints/encoder_pretrain_6v2/best_encoder.pth \
  --episodes 200 --max-steps 1000 --device auto \
  --output checkpoints/end_to_end_6v2_probe \
  --run-name SDGCR_end_to_end_6v2_probe
```
