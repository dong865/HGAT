# 6V2 ROS 2 迁移

This package is the ROS 2 migration of the six-pursuer/two-evader project from
`gym-pybullet-drones-main（6V2）`.  It keeps the original simulation, ST-HGAT,
temporal predictor, Sinkhorn assignment, MADDPG baselines, evaluation programs
and TensorBoard utilities in one importable Python package.

## Paper method (revised route): SDGCR

The revised HGAT paper route is implemented as **Stage-Feasible Dynamic Graph
Coalition Reconfiguration** (`stage_feasible_dgcr`): variable-scale episodes
(Np in {4,5,6}, Ne in {1,2,3}), stage-feasible coalition allocation (masked
Sinkhorn with an idle column + discrete projection with hysteresis/dwell),
a shared velocity Actor and a set-invariant graph Critic, and a task/safety
dual-mask lifecycle (capture -> release -> exit -> landed).  Training,
evaluation, smoke tests and the real-flight runtime are documented in
[README_STAGE.md](README_STAGE.md).

```bash
# train
python -m crazyflie_examples.six_v_two_hgat.main_stage_feasible \
  --episodes 20000 --device auto --output checkpoints/stage_feasible

# evaluate
python -m crazyflie_examples.six_v_two_hgat.evaluate_stage_feasible \
  checkpoints/stage_feasible/best_model.pth

# deploy (6 pursuers + 2 evaders)
python -m crazyflie_examples.six_v_two_hgat.real_flight \
  checkpoints/stage_feasible/best_model.pth \
  --method stage_feasible_dgcr \
  --roles cf1,cf2,cf3,cf4,cf5,cf6,cf7,cf8 \
  --obstacles-file src/crazyswarm2/crazyflie_examples/config/virtual_obstacles_6v2.json
```

## Legacy route

The pre-revision route (fixed 6V2, strict 3+3, ST-HGAT + adaptive cross-group
policy consensus MADDPG) is kept for the five comparison baselines and legacy
checkpoints: `main_parameter_sharing` (legacy proposed method), and the
baselines `main_grouped_maddpg`, `main_independent_maddpg`,
`main_no_graph_maddpg`, `main_role_based_training`, `main_unified_training`.
`real_flight --method legacy` still loads those checkpoints.

## Structural tests

```bash
python -m crazyflie_examples.six_v_two_hgat.smoke_test          # legacy modules
python -m crazyflie_examples.six_v_two_hgat.smoke_test_stage    # SDGCR modules
```

## Monitor training

```bash
tensorboard --logdir runs
```

## Deployment notes

The first six `--roles` names are pursuers; the final two are physical
evaders.  Verify radio channels, rigid-body names, flight volume, separation
thresholds, and an operator emergency stop before propellers are installed.
