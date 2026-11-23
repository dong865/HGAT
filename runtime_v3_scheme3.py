"""Deployment loader for scheme 3.  The offline value teacher is never loaded."""
from pathlib import Path
import numpy as np
import torch
from .stage_networks import StageFeatureEncoder, SharedActor
from .stage_env import StageCoalitionEnv
from .v2_controller import v2_obs_dim
from .v3_demand_distillation import NormalizedDemandHead, V3_DEMAND_SCHEMA
from .v3_scheme3_controller import Scheme3Controller

class Scheme3Runtime:
    def __init__(self, distilled_checkpoint, device="cpu", scenario_seed=None, **kwargs):
        self.device=torch.device(device)
        d=torch.load(Path(distilled_checkpoint),map_location=self.device,weights_only=False)
        if d.get("architecture_version")!=V3_DEMAND_SCHEMA: raise ValueError("not a scheme-3 demand checkpoint")
        p=torch.load(d["source_checkpoint"],map_location=self.device,weights_only=False)
        h=int(d["hidden_dim"]); no=int(p.get("num_obstacles",10)); L=float(p.get("map_size",8.0))
        self.env=StageCoalitionEnv(length=L,num_obstacle=no,max_pursuers=6,max_evaders=3,
                                   d_cap=float(p.get("d_cap",1.5)),hold_steps=int(p.get("hold_steps",5)))
        self.env.reset(num_pursuers=4,num_evaders=2,scenario_seed=scenario_seed)
        enc=StageFeatureEncoder(6,3,no,L,h,10,10,9,self.device,use_ring_slot_features=False,
             use_stage_embedding=False,use_role_embedding=False,use_graph_embedding=True).to(self.device)
        enc._skip_legacy_plan=True; enc.st_hgat.load_state_dict(p["st_hgat"]); enc.relation_head.load_state_dict(p["relation_head"])
        demand=NormalizedDemandHead(h).to(self.device); demand.load_state_dict(d["demand_head"])
        actor=SharedActor(v2_obs_dim(6,h,10),hidden_dim=64).to(self.device); actor.load_state_dict(p["actor"])
        enc.eval(); demand.eval(); actor.eval()
        self.controller=Scheme3Controller(enc,demand,actor,self.device,**kwargs)
        self.loaded_modules=["st_hgat","relation_head","demand_head","actor"]
        self.value_teacher_loaded=False

    def reset(self,scenario_seed=None):
        if scenario_seed is not None:self.env.reset(num_pursuers=4,num_evaders=2,scenario_seed=int(scenario_seed))
        self.controller.reset()

    def step(self, positions_world, velocities_world, origin=(0.0, 0.0)):
        positions=np.asarray(positions_world,dtype=np.float32); velocities=np.asarray(velocities_world,dtype=np.float32)
        n=self.env.num_pursuers+self.env.num_evaders
        if positions.shape not in {(n,2),(n,3)}: raise ValueError("positions shape mismatch")
        local=np.zeros((n,3),dtype=np.float32); local[:,:2]=positions[:,:2]-np.asarray(origin,dtype=np.float32)[:2]+self.env.length/2
        local[:,2]=positions[:,2] if positions.shape[1]==3 else self.env.flight_height
        self.env.positions[:n]=local; self.env.velocities[:n,:2]=velocities[:,:2]; self.env.episode_step_count+=1
        actions,info=self.controller.controller_step(self.env)
        command=np.zeros((n,2),dtype=np.float32); command[:self.env.num_pursuers]=np.clip(actions,-1,1)*float(getattr(self.env,"v_max",0.35))
        return command,info
