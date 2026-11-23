"""Scheme-3 online controller: distilled demand only, no value teacher."""
from __future__ import annotations
import numpy as np
import torch
from .v2_controller import V2Controller
from .v2_env_adapter import apply_membership, capture_hold_active, membership_validity
from .v3_allocation import form_demand_guided_alliance
from .v3_demand_distillation import build_demand_geometry, capacities_from_demand

ARCHITECTURE_VERSION = "normalized_demand_student_v3"

class Scheme3Controller(V2Controller):
    def __init__(self, encoder, demand_head, actor, device, demand_ema=0.9,
                 utility_margin=0.1, min_dwell_steps=50, assign_interval=10, **kwargs):
        super().__init__(encoder, demand_head, actor, device,
                         demand_source="fixed", fixed_demand=3.0,
                         min_dwell_steps=min_dwell_steps,
                         assign_interval=assign_interval, **kwargs)
        self.normalized_demand_head = demand_head
        self.demand_ema = float(demand_ema)
        self.utility_margin = float(utility_margin)

    def reset(self):
        super().reset(); self._rho_ema = None; self._utility = float("-inf")

    @staticmethod
    def _score(costs, assignment, capacity, previous, switch_penalty=0.5):
        a=np.asarray(assignment); c=costs.detach().cpu().numpy(); cap=capacity.detach().cpu().numpy()
        counts=np.bincount(a[a>=0], minlength=len(cap))
        relation=sum(c[i,j] for i,j in enumerate(a) if j>=0)
        switches=sum(int(i<len(previous) and previous[i]!=j) for i,j in enumerate(a))
        return -float(relation)-float(((counts-cap)**2).sum())-switch_penalty*switches

    def controller_step(self, env):
        env.advance_lifecycle(); snap=env.snapshot(); n=env.num_pursuers; m=env.num_evaders
        enc=self.encoder.encode(snap, differentiable=False)
        active=torch.as_tensor(env.task_mask[:m], dtype=torch.bool, device=self.device)
        positions=np.asarray(snap["positions"]); geom=build_demand_geometry(
            positions[:n,:2], positions[n:n+m,:2], self.encoder.map_size).to(self.device)
        with torch.no_grad():
            rho=self.normalized_demand_head(enc["task_embeddings"][:m], active, n, 3, geom)
        if self._rho_ema is None or self._rho_ema.shape != rho.shape: self._rho_ema=rho
        else: self._rho_ema=self.demand_ema*self._rho_ema+(1-self.demand_ema)*rho
        cap=capacities_from_demand(self._rho_ema, active, n)
        ids=torch.nonzero(active).squeeze(-1)
        reconfigured=False; reason="period_pending"
        invalid=not bool(membership_validity(env,self._assignment).all())
        if len(ids)==1: candidate=np.full(n,int(ids[0]),dtype=np.int64)
        elif len(ids)>1:
            local=form_demand_guided_alliance(enc["relation_costs"][:n,ids],cap[ids])
            candidate=ids.detach().cpu().numpy()[local]
        else: candidate=np.full(n,-1,dtype=np.int64)
        utility=self._score(enc["relation_costs"][:n,:m],candidate,cap,self._assignment)
        due=self._step%self.assign_interval==0
        forced=invalid or np.all(self._assignment<0) or len(ids)<=1
        if forced or (due and self._dwell>=self.min_dwell_steps and not capture_hold_active(env)
                      and utility>self._utility+self.utility_margin):
            apply_membership(env,candidate); self._assignment=candidate.copy(); self._utility=utility
            self._dwell=0; reconfigured=True; reason="forced" if forced else "utility_improved"
        else: self._dwell+=1
        enc2=self.encoder.encode(env.snapshot(), differentiable=False)
        obs,_=self._build_v2_obs(enc2,self._assignment,env)
        with torch.no_grad(): actions=self.actor(obs)[:n].cpu().numpy()
        self._step+=1
        return actions,{"architecture_version":ARCHITECTURE_VERSION,"demand":rho.cpu().tolist(),
                        "demand_ema":self._rho_ema.cpu().tolist(),"capacity":cap.cpu().tolist(),
                        "assignment":self._assignment[:n].tolist(),"reconfigured":reconfigured,
                        "gate_reason":reason,"utility":utility}
