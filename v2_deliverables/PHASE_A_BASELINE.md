# Phase A: 版本与协议核实记录
日期: 2026-11-09 21:56:52
仓库路径: /media/yudong/data/yuzhew/ros2_ws_6V2/src/crazyswarm2/crazyflie_examples/crazyflie_examples/six_v_two_hgat

## 核心文件 SHA-256 与行数
ca4c9b08b9e64b5b  stage_networks.py (991 lines)
652b12c35c543393  stage_coalitions.py (1687 lines)
b6bc043ea6998c17  stage_env.py (1602 lines)
096b634fe3a24276  end_to_end_training.py (446 lines)
e76427c139d88edc  main_stage_end_to_end.py (671 lines)
4b470735713e5e24  runtime_stage.py (449 lines)
f0c6400539416449  encoder_pretraining.py (106 lines)
3fa3043b25f9247f  pretrain_stage_encoder.py (245 lines)
f07dc32a4ddd0029  verify_e2e_gradients.py (212 lines)
08ca80536b30b6fa  training_protocol.py (87 lines)
d083ba4e91cadc42  st_hgat.py (491 lines)

## allocate() 调用点
./main_stage_end_to_end.py:492:                allocation = allocator.allocate(
./main_stage_feasible.py:5:    allocator.allocate(geometry, predictions) -> (X, stages)     # stage-capacity-member
./main_stage_feasible.py:468:                alloc = allocator.allocate(
./main_stage_feasible.py:1045:                alloc = allocator.allocate(
./runtime_stage.py:379:            self._alloc = self.allocator.allocate(

## encode() 调用点（非定义）
./diagnose_checkpoint.py:83:                feats = encoder.encode(snap, differentiable=True)
./end_to_end_training.py:155:    current = encoder.encode(
./end_to_end_training.py:162:        following = encoder.encode(
./evaluate_stage_baseline.py:332:    feats = model["encoder"].encode(snap2)
./main_stage_end_to_end.py:489:                    allocation_features = encoder.encode(
./main_stage_end_to_end.py:511:            features = encoder.encode(snapshot)
./main_stage_feasible.py:7:    encoder.encode(snapshot) -> obs                              # padded features
./main_stage_feasible.py:485:            features = encoder.encode(snap)
./main_stage_feasible.py:1063:            features = encoder.encode(snap)
./main_stage_feasible.py:1077:            features_next = encoder.encode(snap_next)
./pretrain_stage_encoder.py:193:            feats = encoder.encode(env.snapshot())
./runtime_stage.py:395:            features = self.encoder.encode(snap)
./verify_e2e_gradients.py:136:    enc_a = encoder.encode(snap, differentiable=True)
./verify_e2e_gradients.py:154:        enc_b = encoder.encode(snap, differentiable=True)
./verify_e2e_gradients.py:164:    enc_c = encoder.encode(snap, differentiable=True)

## apply_roles/apply_assignment 调用点
./evaluate_stage_baseline.py:327:    env.apply_roles(
./main_stage_end_to_end.py:506:            env.apply_roles(
./main_stage_feasible.py:6:    env.apply_assignment(X, stages)                              # apply coalition
./main_stage_feasible.py:477:            env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
./main_stage_feasible.py:1054:            env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
./runtime_stage.py:389:        self.env.apply_roles(alloc["core_assignment"], alloc["sup_assignment"],
./stage_env.py:530:        self.apply_roles(core, sup, roles, stages)

## soft_task_plan 传入点
./main_stage_end_to_end.py:502:                    soft_task_plan=allocation_features["soft_assignment"][
./stage_coalitions.py:830:        soft_task_plan: Optional[torch.Tensor] = None,
./stage_coalitions.py:903:        if soft_task_plan is not None and self.solver_mode == "sinkhorn":
./stage_coalitions.py:905:                soft_task_plan, dtype=torch.float32, device=self.device,
./stage_coalitions.py:909:                    f"soft_task_plan shape {tuple(plan_tensor.shape)} != {(n_p, n_ev)}"
