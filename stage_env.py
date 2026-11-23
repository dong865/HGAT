"""Stage-feasible coalition environment for variable-scale multi-target pursuit.

Implements the environment side of the revised HGAT paper route
(Stage-Feasible Dynamic Graph Coalition Reconfiguration):

* variable episode scale:  Np in {4,5,6} pursuers, Ne in {1,2,3} evaders
* target lifecycle:        active -> captured -> exit -> landed
* dual masks:              task_mask (removed immediately on capture),
                           safe_mask  (kept until the evader lands)
* capture feasibility:     >= 3 pursuers, max distance <= d_cap,
                           evader inside the pursuer polygon (convex hull),
                           held for T_hold steps
* pursuer action:          normalized planar *velocity* command
                           (paper: a_i = [vx, vy]), matching cmdVelocityWorld
* normalized team reward:  r_t = (1/|P_t|) sum_i r_i^P
                           + (1/max(1,|E_task|)) sum_j r_j^E
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .scenario_protocol_v4_dispersed import (
    GoalDirectedPotentialField,
    GoalDirectedScenarioSampler,
    ScenarioConfig,
    goal_reached,
)
from .training_protocol import DEFAULT_D_CAP, DEFAULT_HOLD_STEPS

@dataclass(frozen=True)
class StageRewardConfig:
    time_penalty: float = -0.02
    distance_weight: float = 0.8
    heading_weight: float = 0.01
    gap_weight: float = 1.0
    hold_weight: float = 6.0
    feasibility_weight: float = 0.5
    # dense credit for contracting an assigned coalition's max distance while
    # inside the closure band (teaches squeezing inside d_cap); 0.0 = off
    closure_weight: float = 0.0
    # band (meters) inside which closure contraction is rewarded
    closure_band: float = 3.0
    # Potential-difference reward that directly contracts the maximum angular
    # gap toward the strict enclosure boundary (Psi < pi).  Disabled by default
    # so historical experiments remain reproducible.
    angular_closure_weight: float = 0.0
    angular_closure_band: float = 2.5
    # IBR-style uniform-ring shaping: reward converging to an evenly spaced
    # ring (radius, radial deviation, angular uniformity) around the *looked
    # ahead* evader position. 0.0 = off.
    ring_weight: float = 0.0
    ring_radius: float = 1.5
    ring_lookahead: float = 0.4
    effort_weight: float = 0.02
    smoothness_weight: float = 0.05
    capture_bonus: float = 600.0
    all_captured_bonus: float = 300.0
    hold_weight: float = 30.0
    progress_clip: float = 0.10
    gap_clip: float = 1.0
    evader_goal_penalty: float = -60.0
    timeout_penalty: float = -5.0
    collision_penalty: float = -5.0
    pursuer_collision_penalty: float = -5.0
    obstacle_penalty: float = -2.0
    predictive_boundary_weight: float = 1.0
    predictive_boundary_margin: float = 0.35

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def normalized_progress(previous: float, current: float, scale: float, clip: float) -> float:
    """Positive iff a minimized stage quantity improved."""
    if not np.isfinite(previous) or not np.isfinite(current) or scale <= 0.0:
        return 0.0
    return float(np.clip((previous - current) / scale, -clip, clip))


def enclosure_geometry_score(
    pursuer_positions: np.ndarray,
    evader_position: np.ndarray,
    d_cap: float,
) -> float:
    """Return the best three-agent enclosure score in [0, 1].

    The score combines radial proximity, 120-degree angular balance and
    centroid alignment. If core and support provide more than three
    candidates, the best triplet defines the shaping potential.
    """
    points = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)
    evader = np.asarray(evader_position, dtype=np.float64).reshape(3)
    if len(points) < 3 or d_cap <= 0.0:
        return 0.0

    best = 0.0
    ideal_gap = 2.0 * np.pi / 3.0
    sigma = np.pi / 6.0
    for ids in combinations(range(len(points)), 3):
        tri = points[list(ids)]
        distances = np.linalg.norm(tri - evader, axis=1)
        radial = float(np.exp(-np.max(distances) / d_cap))
        angles = np.sort(np.arctan2(tri[:, 1] - evader[1], tri[:, 0] - evader[0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        angular = float(np.mean(np.exp(-0.5 * ((gaps - ideal_gap) / sigma) ** 2)))
        centroid = float(np.exp(-np.linalg.norm(np.mean(tri, axis=0) - evader) / d_cap))
        best = max(best, radial * (0.7 * angular + 0.3 * centroid))
    return float(np.clip(best, 0.0, 1.0))

MODE_TRK = 0
MODE_BLK = 1
MODE_CAP = 2
MODE_EXIT = 3
MODE_LANDED = 4

MODE_NAMES = {
    MODE_TRK: "trk",
    MODE_BLK: "blk",
    MODE_CAP: "cap",
    MODE_EXIT: "exit",
    MODE_LANDED: "landed",
}

ROLE_CORE = 0
ROLE_SUP = 1
ROLE_RES = 2

ROLE_NAMES = {ROLE_CORE: "core", ROLE_SUP: "support", ROLE_RES: "reserve"}


def mode_capacity(mode: int) -> int:
    """Minimum number of *core* members required by a task stage (paper kappa(m)).

    It is a lower bound on the core role count, not the total coalition size and
    not the number of pursuers allowed to move.
    """
    return {
        MODE_TRK: 1,
        MODE_BLK: 2,
        MODE_CAP: 3,
        MODE_EXIT: 0,
        MODE_LANDED: 0,
    }[mode]


def escape_gap(points: np.ndarray, center: np.ndarray) -> float:
    """Maximum angular gap (rad) of ``points`` around ``center`` (paper eq. Psi_j).

    Returns ``np.inf`` when fewer than one point is provided.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(center, dtype=np.float64).reshape(2)
    if points.shape[0] == 0:
        return float("inf")
    angles = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    k = angles.shape[0]
    diffs = np.empty(k)
    for i in range(k):
        d = angles[(i + 1) % k] - angles[i]
        if d < 0.0:
            d += 2.0 * np.pi
        diffs[i] = d
    return float(np.max(diffs))


def angular_enclosure_potential(
    pursuer_positions: np.ndarray,
    evader_position: np.ndarray,
    neighborhood: float,
) -> float:
    """Best three-agent max-gap potential in [0, 1] inside ``neighborhood``.

    The potential is zero for fewer than three nearby pursuers and reaches one
    exactly when the best triplet's maximum angular gap is at most pi.  Using a
    potential difference rewards angular progress without granting a perpetual
    positive reward for merely holding a formation.
    """
    points = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)
    evader = np.asarray(evader_position, dtype=np.float64).reshape(3)
    if neighborhood <= 0.0:
        return 0.0
    distances = np.linalg.norm(points - evader, axis=1)
    nearby = points[distances <= float(neighborhood)]
    if len(nearby) < 3:
        return 0.0
    best = 0.0
    for ids in combinations(range(len(nearby)), 3):
        psi = escape_gap(nearby[list(ids), :2], evader[:2])
        value = np.clip((2.0 * np.pi - psi) / np.pi, 0.0, 1.0)
        best = max(best, float(value))
    return best


def angular_potential_reward(previous: float, current: float, weight: float) -> float:
    """Unclipped potential difference; closed state loops sum to zero."""
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0
    return float(weight) * (float(current) - float(previous))


def ring_uniformity_potential(
    pursuer_positions: np.ndarray,
    evader_position: np.ndarray,
    evader_velocity: np.ndarray,
    radius: float,
    lookahead: float,
) -> float:
    """Uniform-ring potential in [0, 1] (IBR-style formation prior).

    Center = evader position + lookahead * evader velocity (aims ahead of the
    escape direction). Members within the given set that are >= 3 contribute:
      - radial deviation: mean |r_i - radius| / radius
      - angular deviation: max_i |gap_i - 2*pi/k| / (2*pi/k)
    Potential = 1 - 0.5 * clip(rad,0,1) - 0.5 * clip(ang,0,1).
    Fewer than 3 members -> 0.0 (can't form a ring/enclosure yet).
    """
    points = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)
    evader = np.asarray(evader_position, dtype=np.float64).reshape(3)
    velocity = np.asarray(evader_velocity, dtype=np.float64).reshape(3)
    if len(points) < 3 or radius <= 0.0:
        return 0.0
    center = evader[:2] + float(lookahead) * velocity[:2]
    dxy = points[:, :2] - center[None, :]
    radii = np.linalg.norm(dxy, axis=1)
    rad_err = float(np.mean(np.abs(radii - radius))) / radius
    # angular gaps of the (up to all) points around the center
    angles = np.sort(np.arctan2(dxy[:, 1], dxy[:, 0]))
    k = len(angles)
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    uniform = 2.0 * np.pi / k
    ang_err = float(np.max(np.abs(gaps - uniform))) / uniform
    return float(
        1.0
        - 0.5 * float(np.clip(rad_err, 0.0, 1.0))
        - 0.5 * float(np.clip(ang_err, 0.0, 1.0))
    )


def gap_containing(points: np.ndarray, center: np.ndarray, direction: np.ndarray) -> float:
    """Angular gap (rad) that contains ``direction``, or ``np.inf`` if no point."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(center, dtype=np.float64).reshape(2)
    if points.shape[0] == 0:
        return float("inf")
    ang = float(np.arctan2(direction[1], direction[0]))
    angles = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    k = angles.shape[0]
    for i in range(k):
        lo, hi = angles[i], angles[(i + 1) % k]
        if hi < lo:
            hi += 2.0 * np.pi
        a = ang
        if a < lo:
            a += 2.0 * np.pi
        if lo <= a < hi:
            return float(hi - lo)
    return float("inf")


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """2-D convex hull via monotone chain; returns vertices in CCW order."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting point-in-polygon test (works for convex or concave)."""
    p = np.asarray(point, dtype=np.float64).reshape(2)
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    x, y = p
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def evader_enclosed_by_coalition(pursuer_positions: np.ndarray, evader_position: np.ndarray) -> bool:
    """True iff the evader lies inside the convex hull of the coalition (XY plane)."""
    pts = np.asarray(pursuer_positions, dtype=np.float64).reshape(-1, 3)
    evader = np.asarray(evader_position, dtype=np.float64).reshape(3)
    if len(pts) < 3:
        return False
    hull = _convex_hull(pts[:, :2])
    return _point_in_polygon(evader[:2], hull)


class _Obstacle:
    __slots__ = ("position", "radius", "height")

    def __init__(self, position: np.ndarray, radius: float, height: float = 3.0):
        self.position = position.astype(np.float64)
        self.radius = float(radius)
        self.height = float(height)


class StageCoalitionEnv:
    """Variable-scale 3-D pursuit environment with a stage lifecycle (fixed-height flight)."""

    def __init__(
        self,
        length: float = 8.0,
        num_obstacle: int = 10,
        max_pursuers: int = 6,
        max_evaders: int = 3,
        d_cap: float = DEFAULT_D_CAP,
        hold_steps: int = DEFAULT_HOLD_STEPS,
        v_max: float = 0.35,  # aligned with SixVTwoConfig.max_pursuer_speed
        v_max_e: float = 0.35,
        collision_distance: float = 0.4,
        agent_radius: float = 0.2,
        exit_radius: float = 0.15,
        time_step: float = 1.0 / 50.0,
        flight_height: float = 1.0,
        height_kp: float = 5.0,
        height_kd: float = 1.0,
        height_noise: float = 0.02,
        world_height: float = 2.5,
        obstacle_min_gap: float = 0.35,
        reward_config: Optional[StageRewardConfig] = None,
        ablation_reserve_hover: bool = False,
        perception_radius: Optional[float] = None,
        seed: Optional[int] = None,
        land_in_place: bool = True,
    ):
        self.length = float(length)
        self.num_obstacle = int(num_obstacle)
        self.max_pursuers = int(max_pursuers)
        self.max_evaders = int(max_evaders)
        self.max_agents = self.max_pursuers + self.max_evaders
        self.d_cap = float(d_cap)
        self.hold_steps = int(hold_steps)
        self.v_max = float(v_max)
        self.v_max_e = float(v_max_e)
        # 捕获后的逃逸机行为：True = 原地降落（捕获即记录当前位置为离场点，
        # 下一步转 LANDED 并垂直降落）；False = 旧行为（飞最远角落后降落）。
        self.land_in_place = bool(land_in_place)
        self.collision_distance = float(collision_distance)
        self.agent_radius = float(agent_radius)
        self.exit_radius = float(exit_radius)
        self.time_step = float(time_step)
        self.flight_height = float(flight_height)
        self.height_kp = float(height_kp)
        self.height_kd = float(height_kd)
        self.height_noise = float(height_noise)
        self.world_height = float(world_height)
        self.obstacle_min_gap = float(obstacle_min_gap)
        self.reward_config = reward_config or StageRewardConfig()
        # training-time difficulty scale for evader dynamics (1.0 = protocol
        # speed); evaluation always forces 1.0 via evaluate()/step callers
        self.evader_speed_scale = 1.0
        # per-evader previous coalition max member distance (closure shaping cache)
        self.prev_member_max = np.full(self.max_evaders, np.inf, dtype=np.float64)
        self.ablation_reserve_hover = bool(ablation_reserve_hover)
        # None = full observability (default, preserves prior behaviour); a finite
        # value restricts each pursuer's perception to entities within this radius.
        self.perception_radius = None if perception_radius is None else float(perception_radius)
        self._root_seed = int(seed or 0)
        self._episode_index = 0
        self._rng = np.random.default_rng(self._root_seed)
        self.scenario_config = ScenarioConfig(
            flight_height=self.flight_height, length=self.length,
            num_obstacles=self.num_obstacle, agent_radius=self.agent_radius,
            collision_distance=self.collision_distance, max_speed=self.v_max_e,
            capture_radius=self.d_cap,
        )
        self.scenario_sampler = GoalDirectedScenarioSampler(self.scenario_config)
        self.evader_field = GoalDirectedPotentialField(self.scenario_config)

        self.obstacles: List[_Obstacle] = []

        # Episode scale and state buffers (padded to max sizes).
        self.num_pursuers = 0
        self.num_evaders = 0
        self.positions = np.zeros((self.max_agents, 3), dtype=np.float64)
        self.velocities = np.zeros((self.max_agents, 3), dtype=np.float64)
        self.pursuer_mask = np.zeros(self.max_pursuers, dtype=bool)
        self.task_mask = np.zeros(self.max_evaders, dtype=bool)
        self.safe_mask = np.zeros(self.max_evaders, dtype=bool)
        self.stages = np.zeros(self.max_evaders, dtype=np.int64)
        self.captured = np.zeros(self.max_evaders, dtype=bool)
        self.hold_counters = np.zeros(self.max_evaders, dtype=np.int64)
        self.prev_gaps = np.full(self.max_evaders, np.inf, dtype=np.float64)
        self.prev_geometry_scores = np.zeros(self.max_evaders, dtype=np.float64)
        self.prev_angular_potentials = np.zeros(self.max_evaders, dtype=np.float64)
        self.prev_ring_potentials = np.zeros(self.max_evaders, dtype=np.float64)
        self.prev_pair_distances = np.full(
            (self.max_pursuers, self.max_evaders), np.inf, dtype=np.float64,
        )
        self.exit_targets = np.zeros((self.max_evaders, 3), dtype=np.float64)
        self.evader_goals = np.zeros((self.max_evaders, 3), dtype=np.float64)
        self.scenario_seed = 0
        self.outcome = "running"
        # operational roles: ROLE_CORE / ROLE_SUP / ROLE_RES (every active pursuer)
        self.roles = np.full(self.max_pursuers, ROLE_RES, dtype=np.int64)
        self.prev_roles = np.full(self.max_pursuers, ROLE_RES, dtype=np.int64)
        self.assignment = -np.ones(self.max_pursuers, dtype=np.int64)
        self.prev_assignment = -np.ones(self.max_pursuers, dtype=np.int64)
        self.prev_distances = np.zeros(self.max_pursuers, dtype=np.float64)
        self.prev_commands = np.zeros((self.max_pursuers, 2), dtype=np.float64)
        self._shaping_signatures = [tuple() for _ in range(self.max_evaders)]
        self.capture_coalitions: Dict[int, List[int]] = {}
        self.episode_step_count = 0
        self.done = False

    # ------------------------------------------------------------------ setup
    def _sample_obstacle(self) -> _Obstacle:
        margin = 0.5
        radius = float(self._rng.uniform(0.12, 0.22))
        for _ in range(200):
            pos = self._rng.uniform(margin, self.length - margin, size=2)
            ok = True
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs.position) < obs.radius + radius + self.obstacle_min_gap:
                    ok = False
                    break
            if ok:
                return _Obstacle(pos, radius)
        return _Obstacle(np.array([self.length / 2.0, self.length / 2.0]), radius)

    def sample_scale(self) -> Tuple[int, int]:
        """Sample an episode scale from the paper protocol ranges."""
        return int(self._rng.choice([4, 5, 6])), int(self._rng.choice([1, 2, 3]))

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        num_pursuers: Optional[int] = None,
        num_evaders: Optional[int] = None,
        scenario_seed: Optional[int] = None,
    ) -> "StageCoalitionEnv":
        if num_pursuers is None or num_evaders is None:
            num_pursuers, num_evaders = self.sample_scale()
        num_pursuers = int(np.clip(num_pursuers, 3, self.max_pursuers))
        num_evaders = int(np.clip(num_evaders, 1, self.max_evaders))
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders

        self.positions.fill(0.0)
        self.velocities.fill(0.0)
        self.pursuer_mask[:] = False
        self.pursuer_mask[:num_pursuers] = True
        self.task_mask[:] = False
        self.task_mask[:num_evaders] = True
        self.safe_mask[:] = False
        self.safe_mask[:num_evaders] = True
        self.stages[:] = MODE_TRK
        self.captured[:] = False
        self.hold_counters[:] = 0
        self.prev_gaps[:] = np.inf
        self.prev_geometry_scores[:] = 0.0
        self.prev_angular_potentials[:] = 0.0
        self.prev_ring_potentials[:] = 0.0
        self.prev_pair_distances[:] = np.inf
        self.prev_member_max[:] = np.inf
        self.exit_targets[:] = 0.0
        self.roles[:] = ROLE_RES
        self.prev_roles[:] = ROLE_RES
        self.assignment[:] = -1
        self.prev_assignment[:] = -1
        self.prev_commands[:] = 0.0
        self._shaping_signatures = [tuple() for _ in range(self.max_evaders)]
        self.capture_coalitions = {}
        self.episode_step_count = 0
        self.done = False
        self.outcome = "running"
        if scenario_seed is None:
            scenario_seed = self._root_seed + self._episode_index
        self._episode_index += 1
        self.scenario_seed = int(scenario_seed)
        # The scenario seed governs both initial geometry and all stochastic
        # dynamics in the episode.  A local generator prevents evaluation
        # results from depending on unrelated global NumPy draws.
        self._rng = np.random.default_rng(self.scenario_seed)
        scene = self.scenario_sampler.sample(num_pursuers, num_evaders, self.scenario_seed)
        self.positions[:num_pursuers, :2] = scene.pursuer_positions[:, :2]
        self.positions[:num_pursuers, 2] = self.flight_height
        self.positions[num_pursuers:num_pursuers + num_evaders, :2] = scene.evader_positions[:, :2]
        self.positions[num_pursuers:num_pursuers + num_evaders, 2] = self.flight_height
        self.evader_goals.fill(0.0)
        self.evader_goals[:num_evaders, :2] = scene.evader_goals[:, :2]
        self.evader_goals[:num_evaders, 2] = self.flight_height
        self.obstacles = [_Obstacle(o[:2], o[2], o[3]) for o in scene.obstacles]

        # Initial roles: every pursuer starts as a core member of its nearest evader.
        for i in range(self.num_pursuers):
            self.assignment[i] = self._nearest_evader(i)
        self.roles[:num_pursuers] = ROLE_CORE
        self._refresh_reward_baselines()
        return self

    # ------------------------------------------------------------- assignment
    def _nearest_evader(self, pursuer_idx: int) -> int:
        best, best_d = 0, float("inf")
        p = self.positions[pursuer_idx]
        for j in range(self.num_evaders):
            if not self.task_mask[j]:
                continue
            d = float(np.linalg.norm(p - self.positions[self.num_pursuers + j]))
            if d < best_d:
                best_d, best = d, j
        return best

    def apply_assignment(self, assignment: Sequence[int], stages: Optional[Sequence[int]] = None) -> None:
        """Backward-compatible single-role apply: all members are core members."""
        core = list(assignment)
        sup = [-1] * self.num_pursuers
        roles = [ROLE_CORE if int(a) >= 0 else ROLE_RES for a in assignment]
        self.apply_roles(core, sup, roles, stages)

    def apply_roles(
        self,
        core_assignment: Sequence[int],
        sup_assignment: Sequence[int],
        roles: Sequence[int],
        stages: Optional[Sequence[int]] = None,
    ) -> None:
        """Apply core/support/reserve roles (target id per member, -1 otherwise).

        Every active pursuer holds exactly one role; the reserve role is a
        *mobile* role and never means "stop moving".
        """
        old_signatures = [
            tuple(self.shaping_members(j)) for j in range(self.num_evaders)
        ]
        for i in range(self.num_pursuers):
            role = int(roles[i])
            target = -1
            if role == ROLE_CORE:
                target = int(core_assignment[i])
            elif role == ROLE_SUP:
                target = int(sup_assignment[i])
            if target < 0 or target >= self.num_evaders or not self.task_mask[target]:
                self.roles[i] = ROLE_RES
                self.assignment[i] = -1
            else:
                self.roles[i] = role
                self.assignment[i] = target
        if stages is not None:
            for j in range(self.num_evaders):
                if self.task_mask[j] and not self.captured[j]:
                    self.stages[j] = int(stages[j])
        current_pairs = self._pairwise_distances()
        self._refresh_prev_distances(current_pairs)
        for j in range(self.num_evaders):
            signature = tuple(self.shaping_members(j))
            if signature != old_signatures[j] or signature != self._shaping_signatures[j]:
                members = list(signature)
                self.prev_gaps[j] = self._shaping_gap(j)
                self.prev_geometry_scores[j] = self._geometry_score(j)
                self.prev_angular_potentials[j] = angular_enclosure_potential(
                    self.positions[members],
                    self.positions[self.num_pursuers + j],
                    self.reward_config.angular_closure_band,
                )
                self.prev_ring_potentials[j] = ring_uniformity_potential(
                    self.positions[members],
                    self.positions[self.num_pursuers + j],
                    self.evader_velocities()[j],
                    self.reward_config.ring_radius,
                    self.reward_config.ring_lookahead,
                )
                # membership changed: reset closure baseline to avoid a false
                # "contraction" credit across a different member set
                if members:
                    evader_pos = self.positions[self.num_pursuers + j]
                    self.prev_member_max[j] = float(
                        np.max(np.linalg.norm(self.positions[members] - evader_pos, axis=1))
                    )
                else:
                    self.prev_member_max[j] = np.inf
            self._shaping_signatures[j] = signature

    def core_members(self, evader_idx: int) -> List[int]:
        return [
            i for i in range(self.num_pursuers)
            if self.roles[i] == ROLE_CORE and self.assignment[i] == evader_idx
        ]

    def support_members(self, evader_idx: int) -> List[int]:
        return [
            i for i in range(self.num_pursuers)
            if self.roles[i] == ROLE_SUP and self.assignment[i] == evader_idx
        ]

    def reserve_members(self) -> List[int]:
        return [i for i in range(self.num_pursuers) if self.roles[i] == ROLE_RES]

    def coalition_of(self, evader_idx: int) -> List[int]:
        """Core members of an evader (the coalition that defines feasibility)."""
        return self.core_members(evader_idx)

    def shaping_members(self, evader_idx: int) -> List[int]:
        """Assigned core and support members that share geometric shaping.

        Capture feasibility remains defined exclusively by ``coalition_of``.
        Support agents receive the same enclosure signal because their role is
        to help create a feasible core formation, while reserve agents remain
        outside the target-specific shaping term.
        """
        return [
            i for i in range(self.num_pursuers)
            if self.roles[i] in (ROLE_CORE, ROLE_SUP)
            and self.assignment[i] == evader_idx
        ]

    def free_pursuers(self) -> List[int]:
        return [i for i in range(self.num_pursuers) if self.assignment[i] < 0]

    # ------------------------------------------------------------ capture
    def _coalition_gap(self, j: int) -> float:
        coalition = self.coalition_of(j)
        if len(coalition) < 2:
            return float("inf")
        return escape_gap(self.positions[coalition][:, :2], self.positions[self.num_pursuers + j, :2])

    def _shaping_gap(self, j: int) -> float:
        members = self.shaping_members(j)
        if len(members) < 2:
            return float("inf")
        return escape_gap(
            self.positions[members][:, :2],
            self.positions[self.num_pursuers + j, :2],
        )

    def _geometry_score(self, j: int) -> float:
        members = self.shaping_members(j)
        if len(members) < 3:
            return 0.0
        return enclosure_geometry_score(
            self.positions[members],
            self.positions[self.num_pursuers + j],
            self.d_cap,
        )

    def _capture_hold_ok(self, j: int) -> bool:
        coalition = self.coalition_of(j)
        if len(coalition) < 3:
            return False
        evader_pos = self.positions[self.num_pursuers + j]
        pts = self.positions[coalition]
        if float(np.max(np.linalg.norm(pts - evader_pos, axis=1))) > self.d_cap:
            return False
        if not evader_enclosed_by_coalition(pts, evader_pos):
            return False
        return True

    # ------------------------------------------------------------ dynamics
    def _evader_velocity(self, j: int) -> np.ndarray:
        obstacles = np.asarray(
            [[o.position[0], o.position[1], o.radius] for o in self.obstacles],
            dtype=np.float64,
        ).reshape(-1, 3)  # potential field ignores obstacle height
        vel = self.evader_field.velocity(
            j, self.pursuer_positions(), self.evader_positions(),
            self.evader_goals[:self.num_evaders], obstacles,
        )
        # training-time difficulty curriculum: slow the evasion dynamics by a
        # factor (1.0 = protocol). Evaluation always uses 1.0.
        return vel * self.evader_speed_scale

    def _exit_velocity(self, j: int) -> np.ndarray:
        pos = self.positions[self.num_pursuers + j, :2]
        target = self.exit_targets[j, :2]
        diff = target - pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-3:
            return np.zeros(2)
        return diff / dist * min(self.v_max_e, dist / max(self.time_step, 1e-6) * 0.5)

    def _soft_boundary(self, idx: int) -> None:
        soft = self.agent_radius + 0.15
        for dim in range(2):
            if self.positions[idx][dim] < soft:
                if self.velocities[idx][dim] < 0.06:
                    self.velocities[idx][dim] += 0.12
            elif self.positions[idx][dim] > self.length - soft:
                if self.velocities[idx][dim] > -0.06:
                    self.velocities[idx][dim] -= 0.12
        for dim in range(2):
            if self.positions[idx][dim] < self.agent_radius:
                self.positions[idx][dim] = self.agent_radius + 0.01
                self.velocities[idx][dim] = abs(self.velocities[idx][dim]) * 0.8 + 0.08
            elif self.positions[idx][dim] > self.length - self.agent_radius:
                self.positions[idx][dim] = self.length - self.agent_radius - 0.01
                self.velocities[idx][dim] = -abs(self.velocities[idx][dim]) * 0.8 - 0.08

    def _bounce(self) -> None:
        for i in range(self.max_agents):
            if not self._agent_active(i):
                continue
            for obs in self.obstacles:
                horizontal = self.positions[i, :2] - obs.position
                h_dist = float(np.linalg.norm(horizontal))
                cd = obs.radius + self.agent_radius
                z = self.positions[i, 2]
                # cylinder collision: within radius and below obstacle top
                if 1e-6 < h_dist < cd and z < obs.height:
                    normal = np.zeros(3)
                    normal[:2] = horizontal / h_dist
                    self.positions[i, :2] = obs.position + normal[:2] * (cd + 0.01)
                    vn = float(np.dot(self.velocities[i], normal))
                    if vn < 0.0:
                        self.velocities[i] -= 2.0 * vn * normal
                    self.velocities[i] *= 0.8
                    break
        for i in range(self.max_agents):
            if not self._agent_active(i):
                continue
            for k in range(i + 1, self.max_agents):
                if not self._agent_active(k):
                    continue
                diff = self.positions[i] - self.positions[k]
                dist = float(np.linalg.norm(diff))
                if dist < self.collision_distance and dist > 1e-6:
                    normal = diff / dist
                    overlap = self.collision_distance - dist
                    self.positions[i] += normal * (overlap * 0.5 + 0.01)
                    self.positions[k] -= normal * (overlap * 0.5 + 0.01)
                    vn = float(np.dot(self.velocities[i] - self.velocities[k], normal))
                    if vn < 0.0:
                        self.velocities[i] -= 0.5 * vn * normal
                        self.velocities[k] += 0.5 * vn * normal
                    self.velocities[i] *= 0.8
                    self.velocities[k] *= 0.8

    def _detect_contact_events(self) -> Dict[str, np.ndarray]:
        agent_collision = np.zeros(self.max_agents, dtype=bool)
        pursuer_collision = np.zeros(self.max_pursuers, dtype=bool)
        obstacle_collision = np.zeros(self.max_agents, dtype=bool)
        for i in range(self.max_agents):
            if not self._agent_active(i):
                continue
            for k in range(i + 1, self.max_agents):
                if not self._agent_active(k):
                    continue
                if np.linalg.norm(self.positions[i] - self.positions[k]) < self.collision_distance:
                    agent_collision[i] = agent_collision[k] = True
                    if i < self.num_pursuers and k < self.num_pursuers:
                        pursuer_collision[i] = pursuer_collision[k] = True
            for obs in self.obstacles:
                horizontal = np.linalg.norm(self.positions[i, :2] - obs.position)
                if horizontal < obs.radius + self.agent_radius and self.positions[i, 2] < obs.height:
                    obstacle_collision[i] = True
                    break
        return {
            "agent_collision": agent_collision,
            "pursuer_collision": pursuer_collision,
            "obstacle_collision": obstacle_collision,
        }

    def _agent_active(self, idx: int) -> bool:
        if idx < self.num_pursuers:
            return bool(self.pursuer_mask[idx])
        j = idx - self.num_pursuers
        if j >= self.num_evaders:
            return False  # padded slot beyond the current episode scale
        return bool(self.safe_mask[j])  # exiting/active evaders are physical

    # ------------------------------------------------------------ lifecycle
    def advance_lifecycle(self) -> List[int]:
        """Advance the task/safety dual-mask lifecycle from current geometry.

        Returns the list of evaders newly captured in this step.  Used by
        ``step()`` during training and by the real-flight runtime on mocap state.
        """
        capture_events = []
        for j in range(self.num_evaders):
            if self.captured[j]:
                # captured targets still transition exit -> landed (dual mask
                # (0,1) -> (0,0)); they no longer take part in task allocation.
                if self.stages[j] == MODE_EXIT and float(
                    np.linalg.norm(
                        self.positions[self.num_pursuers + j] - self.exit_targets[j]
                    )
                ) <= self.exit_radius:
                    self.stages[j] = MODE_LANDED
                    self.safe_mask[j] = False
                continue
            if self.task_mask[j]:
                if self._capture_hold_ok(j):
                    self.hold_counters[j] += 1
                else:
                    self.hold_counters[j] = 0
                if self.hold_counters[j] >= self.hold_steps:
                    self.captured[j] = True
                    self.task_mask[j] = False
                    self.safe_mask[j] = True  # dual mask transition (1,1)->(0,1)
                    self.stages[j] = MODE_EXIT
                    coalition = self.coalition_of(j)
                    self.capture_coalitions[j] = coalition
                    self.exit_targets[j] = (
                        self.positions[self.num_pursuers + j].copy()
                        if self.land_in_place
                        else self._pick_exit_target(self.positions[self.num_pursuers + j])
                    )
                    # release core and support members into the mobile reserve;
                    # no pursuer stops moving after a release (paper lifecycle).
                    for i in range(self.num_pursuers):
                        if self.assignment[i] == j and self.roles[i] in (ROLE_CORE, ROLE_SUP):
                            self.roles[i] = ROLE_RES
                            self.assignment[i] = -1
                    capture_events.append(j)
        return capture_events

    # ------------------------------------------------------------------ step
    def step(
        self,
        pursuer_velocity_cmd: np.ndarray,
        max_steps: int = 1000,
    ) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        """Advance one 50 Hz step.

        Args:
            pursuer_velocity_cmd: (max_pursuers, 2) normalized planar velocity in [-1, 1].

        Returns:
            rewards (dict), team_reward (float), done (bool), info (dict)
        """
        cmd = np.asarray(pursuer_velocity_cmd, dtype=np.float64).reshape(self.max_pursuers, 2)
        self.prev_assignment = self.assignment.copy()
        reward_assignment = self.assignment.copy()
        reward_roles = self.roles.copy()
        task_mask_before = self.task_mask.copy()
        prev_pair_dists = self.prev_pair_distances.copy()
        prev_gaps = self.prev_gaps.copy()
        prev_geometry = self.prev_geometry_scores.copy()
        prev_angular = self.prev_angular_potentials.copy()
        prev_ring = self.prev_ring_potentials.copy()
        prev_holds = self.hold_counters.copy()
        prev_commands = self.prev_commands.copy()

        # ---- 1. dynamics: every active pursuer moves (core/support/reserve)
        boundary_penalties = np.zeros(self.max_agents, dtype=np.float64)
        for i in range(self.num_pursuers):
            if not self.pursuer_mask[i]:
                continue
            if self.ablation_reserve_hover and self.roles[i] == ROLE_RES:
                self.velocities[i, :2] *= 0.9  # ablation: reserve degrades to hover
            else:
                vel = np.clip(cmd[i], -1.0, 1.0)
                speed = float(np.linalg.norm(vel))
                if speed > 1.0:
                    vel = vel / speed
                self.velocities[i, :2] = vel * self.v_max
            # fixed-height controller with small fluctuation (flight-controller model)
            z_err = self.positions[i, 2] - self.flight_height
            self.velocities[i, 2] += (-self.height_kp * z_err - self.height_kd * self.velocities[i, 2]) * self.time_step
            self.velocities[i, 2] += self._rng.normal(0.0, self.height_noise)
            self.velocities[i, 2] = float(np.clip(self.velocities[i, 2], -1.0, 1.0))
            self.positions[i] += self.velocities[i] * self.time_step
            self.positions[i, 2] = float(np.clip(self.positions[i, 2], self.agent_radius, self.world_height - self.agent_radius))
            boundary_penalties[i] = self._boundary_penalty_at(self.positions[i])
            self._soft_boundary(i)

        for j in range(self.num_evaders):
            idx = self.num_pursuers + j
            if self.captured[j] and self.stages[j] == MODE_EXIT:
                self.velocities[idx, :2] = self._exit_velocity(j)
            elif self.task_mask[j]:
                self.velocities[idx, :2] = self._evader_velocity(j)
            else:
                self.velocities[idx, :2] = np.zeros(2)
            # evader height fluctuates around flight_height too
            if self.stages[j] == MODE_LANDED:
                # 原地降落：xy 速度已为零（captured 且非 EXIT 分支），垂直落地
                self.velocities[idx, 2] = -min(
                    0.5, max(0.0, self.positions[idx, 2] - self.agent_radius) * 2.0
                )
                self.positions[idx] += self.velocities[idx] * self.time_step
                self.positions[idx, 2] = float(np.clip(
                    self.positions[idx, 2], self.agent_radius,
                    self.world_height - self.agent_radius))
                continue
            z_err = self.positions[idx, 2] - self.flight_height
            self.velocities[idx, 2] += (-self.height_kp * z_err - self.height_kd * self.velocities[idx, 2]) * self.time_step
            self.velocities[idx, 2] += self._rng.normal(0.0, self.height_noise)
            self.velocities[idx, 2] = float(np.clip(self.velocities[idx, 2], -1.0, 1.0))
            self.positions[idx] += self.velocities[idx] * self.time_step
            self.positions[idx, 2] = float(np.clip(self.positions[idx, 2], self.agent_radius, self.world_height - self.agent_radius))
            boundary_penalties[idx] = self._boundary_penalty_at(self.positions[idx])
            self._soft_boundary(idx)

        # Contact flags must be captured before the bounce solver separates bodies.
        contact_events = self._detect_contact_events()
        self._bounce()
        current_pair_dists = self._pairwise_distances()

        # Cache pre-release coalition geometry. advance_lifecycle() releases a
        # successful coalition, but that success still belongs to this action.
        reward_members = []
        current_gaps = np.zeros(self.max_evaders, dtype=np.float64)
        current_geometry = np.zeros(self.max_evaders, dtype=np.float64)
        current_angular = np.zeros(self.max_evaders, dtype=np.float64)
        current_ring = np.zeros(self.max_evaders, dtype=np.float64)
        capture_feasible = np.zeros(self.max_evaders, dtype=bool)
        for j in range(self.num_evaders):
            members = [
                i for i in range(self.num_pursuers)
                if reward_assignment[i] == j and reward_roles[i] in (ROLE_CORE, ROLE_SUP)
            ]
            reward_members.append(members)
            current_gaps[j] = self._shaping_gap(j)
            current_geometry[j] = enclosure_geometry_score(
                self.positions[members],
                self.positions[self.num_pursuers + j],
                self.d_cap,
            )
            current_angular[j] = angular_enclosure_potential(
                self.positions[members],
                self.positions[self.num_pursuers + j],
                self.reward_config.angular_closure_band,
            )
            current_ring[j] = ring_uniformity_potential(
                self.positions[members],
                self.positions[self.num_pursuers + j],
                self.evader_velocities()[j],
                self.reward_config.ring_radius,
                self.reward_config.ring_lookahead,
            )
            capture_feasible[j] = bool(task_mask_before[j] and self._capture_hold_ok(j))

        # ---- 2. lifecycle and terminal events caused by the current action
        capture_events = self.advance_lifecycle()
        capture_success = bool(self.all_captured())
        reached_evader = None
        if not capture_success:
            for j in range(self.num_evaders):
                if self.task_mask[j] and goal_reached(
                    self.positions[self.num_pursuers + j],
                    self.evader_goals[j],
                    self.scenario_config.goal_radius,
                ):
                    reached_evader = j
                    break

        # ---- 3. decomposed rewards
        component_names = (
            "time", "collision", "boundary", "predictive_boundary", "obstacle",
            "approach", "heading", "geometry", "hold", "feasibility", "closure",
            "angular_closure", "ring",
            "effort", "smoothness", "capture", "all_capture",
        )
        components = {
            name: np.zeros(self.max_pursuers, dtype=np.float64)
            for name in component_names
        }
        for i in range(self.num_pursuers):
            if not self.pursuer_mask[i]:
                continue  # offline pursuer: no reward, excluded from team average
            components["time"][i] = self.reward_config.time_penalty
            # a pursuer-pursuer contact counts once (pursuer_collision), any
            # other agent contact (e.g. pursuer-evader) counts once (collision);
            # previously a pursuer-pursuer contact fired both -5 penalties (-10).
            if contact_events["pursuer_collision"][i]:
                components["collision"][i] += self.reward_config.pursuer_collision_penalty
            elif contact_events["agent_collision"][i]:
                components["collision"][i] += self.reward_config.collision_penalty
            components["boundary"][i] = boundary_penalties[i]
            components["predictive_boundary"][i] = self._predictive_boundary_penalty(i, cmd)
            if contact_events["obstacle_collision"][i]:
                components["obstacle"][i] = self.reward_config.obstacle_penalty

            target = int(reward_assignment[i])
            if (
                reward_roles[i] in (ROLE_CORE, ROLE_SUP)
                and 0 <= target < self.num_evaders
                and task_mask_before[target]
            ):
                d_prev = float(prev_pair_dists[i, target])
                d_now = float(current_pair_dists[i, target])
                if np.isfinite(d_prev) and np.isfinite(d_now):
                    components["approach"][i] = self._approach_reward(d_prev, d_now)
                components["heading"][i] = (
                    self.reward_config.heading_weight * self._heading_reward(i, target)
                )

            executed = self.velocities[i, :2] / max(self.v_max, 1e-6)
            components["effort"][i] = -self.reward_config.effort_weight * float(
                np.dot(executed, executed)
            )
            delta_cmd = cmd[i] - prev_commands[i]
            components["smoothness"][i] = -self.reward_config.smoothness_weight * float(
                np.dot(delta_cmd, delta_cmd)
            )

        per_evader = np.zeros(self.max_evaders, dtype=np.float64)
        for j in range(self.num_evaders):
            if not task_mask_before[j]:
                continue
            d_prev = self._min_pursuer_dist(j, prev_pair_dists)
            d_now = self._min_pursuer_dist(j, current_pair_dists)
            per_evader[j] = self.reward_config.distance_weight * normalized_progress(
                d_prev, d_now, self.v_max, self.reward_config.progress_clip
            )

            members = reward_members[j]
            if members:
                geometry_delta = float(np.clip(
                    current_geometry[j] - prev_geometry[j],
                    -self.reward_config.progress_clip,
                    self.reward_config.progress_clip,
                ))
                geometry_reward = self.reward_config.gap_weight * (
                    geometry_delta / max(self.reward_config.progress_clip, 1e-6)
                )
                hold_reward = self.reward_config.hold_weight * (
                    float(self.hold_counters[j] - prev_holds[j]) / max(1, self.hold_steps)
                )
                feasible_reward = (
                    self.reward_config.feasibility_weight if capture_feasible[j] else 0.0
                )
                # closure: dense credit for contracting the coalition's max
                # distance while inside the closure band (teaches squeezing
                # inside d_cap under the unchanged capture rule)
                now_member_max = float("inf")
                if len(members) >= 2:
                    now_member_max = float(
                        np.max([current_pair_dists[i, j] for i in members])
                    )
                cw = self.reward_config.closure_weight
                closure_reward = 0.0
                if cw > 0.0 and len(members) >= 2:
                    prev_max = float(self.prev_member_max[j])
                    if now_member_max <= self.reward_config.closure_band and np.isfinite(prev_max):
                        closure_reward = cw * normalized_progress(
                            prev_max, now_member_max,
                            self.v_max, self.reward_config.progress_clip,
                        )
                self.prev_member_max[j] = np.inf if len(members) < 2 else now_member_max
                angular_reward = 0.0
                aw = self.reward_config.angular_closure_weight
                if aw > 0.0 and len(members) >= 3:
                    angular_reward = angular_potential_reward(
                        prev_angular[j], current_angular[j], aw,
                    )
                ring_reward = 0.0
                rw = self.reward_config.ring_weight
                if rw > 0.0 and len(members) >= 3:
                    ring_reward = angular_potential_reward(
                        prev_ring[j], current_ring[j], rw,
                    )
                share = 1.0 / len(members)
                for i in members:
                    components["geometry"][i] += geometry_reward * share
                    components["hold"][i] += hold_reward * share
                    components["feasibility"][i] += feasible_reward * share
                    components["closure"][i] += closure_reward * share
                    components["angular_closure"][i] += angular_reward * share
                    components["ring"][i] += ring_reward * share

        for j in capture_events:
            coalition = self.capture_coalitions[j]
            for i in coalition:
                components["capture"][i] += (
                    self.reward_config.capture_bonus / max(1, len(coalition))
                )
        if self.all_captured():
            for i in range(self.num_pursuers):
                components["all_capture"][i] += (
                    self.reward_config.all_captured_bonus / max(1, self.num_evaders)
                )

        per_pursuer = np.sum(np.stack(list(components.values()), axis=0), axis=0)
        active_tasks = int(max(1, np.count_nonzero(task_mask_before[:self.num_evaders])))
        # team average over *online* pursuers so mid-episode dropouts do not
        # rescale the team reward (dimension-change consistency)
        n_active = max(1, self.active_pursuer_count)
        reward_components = {
            name: float(np.sum(values[:self.num_pursuers]) / n_active)
            for name, values in components.items()
        }
        reward_components["evader_progress"] = float(
            np.sum(per_evader[:self.num_evaders]) / active_tasks
        )

        self.episode_step_count += 1
        if capture_success:
            self.outcome = "all_captured"
        elif reached_evader is not None:
            self.outcome = "evader_goal_success"
            reward_components["terminal"] = self.reward_config.evader_goal_penalty
        elif self.episode_step_count >= max_steps:
            self.outcome = "timeout"
            reward_components["terminal"] = self.reward_config.timeout_penalty
        else:
            self.outcome = "running"
            reward_components["terminal"] = 0.0

        team_reward = float(sum(reward_components.values()))
        done = self.outcome != "running"
        self.done = done

        # Update shaping references only after this transition is fully scored.
        self.prev_pair_distances = current_pair_dists.copy()
        self._refresh_prev_distances(current_pair_dists)
        self.prev_gaps[:] = current_gaps
        self.prev_geometry_scores[:] = current_geometry
        self.prev_angular_potentials[:] = current_angular
        self.prev_ring_potentials[:] = current_ring
        self.prev_commands[:] = cmd
        self._shaping_signatures = [
            tuple(self.shaping_members(j)) for j in range(self.max_evaders)
        ]

        info = {
            "per_pursuer": per_pursuer.copy(),
            "per_evader": per_evader.copy(),
            "reward_components": reward_components,
            "capture_events": capture_events,
            "capture_feasible": capture_feasible.copy(),
            "safety_events": {key: value.copy() for key, value in contact_events.items()},
            "boundary_events": boundary_penalties.copy(),
            "stages": self.stages.copy(),
            "task_masks": self.task_mask.copy(),
            "safe_masks": self.safe_mask.copy(),
            "captured": self.captured.copy(),
            "roles": self.roles.copy(),
            "assignment": self.assignment.copy(),
            "episode_step": self.episode_step_count,
            "team_reward": team_reward,
            "outcome": self.outcome,
            "evader_goal_reached": reached_evader,
            "scenario_seed": self.scenario_seed,
        }
        return team_reward, done, info

    def all_captured(self) -> bool:
        return bool(np.all(self.captured[: self.num_evaders]))

    def _pick_exit_target(self, pos: np.ndarray) -> np.ndarray:
        corners = [
            np.array([self.agent_radius + 0.3, self.agent_radius + 0.3, self.flight_height]),
            np.array([self.length - self.agent_radius - 0.3, self.agent_radius + 0.3, self.flight_height]),
            np.array([self.agent_radius + 0.3, self.length - self.agent_radius - 0.3, self.flight_height]),
            np.array([self.length - self.agent_radius - 0.3, self.length - self.agent_radius - 0.3, self.flight_height]),
        ]
        return corners[int(np.argmax([np.linalg.norm(c - pos) for c in corners]))]

    def _pairwise_distances(self) -> np.ndarray:
        distances = np.full(
            (self.max_pursuers, self.max_evaders), np.inf, dtype=np.float64,
        )
        for i in range(self.num_pursuers):
            if not self.pursuer_mask[i]:
                continue
            for j in range(self.num_evaders):
                distances[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[self.num_pursuers + j]
                )
        return distances

    def _min_pursuer_dist(self, j: int, distances: np.ndarray, use_prev: bool = True) -> float:
        matrix = np.asarray(distances)
        if matrix.ndim == 2:
            values = matrix[:self.num_pursuers, j]
        elif use_prev:
            values = matrix[:self.num_pursuers]
        else:
            values = self._pairwise_distances()[:self.num_pursuers, j]
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if len(finite) else 0.0

    def _refresh_prev_distances(self, pair_distances: Optional[np.ndarray] = None) -> None:
        pairs = self._pairwise_distances() if pair_distances is None else pair_distances
        for i in range(self.num_pursuers):
            target = self.assignment[i]
            if target >= 0 and target < self.num_evaders and self.task_mask[target]:
                self.prev_distances[i] = float(pairs[i, target])
            else:
                self.prev_distances[i] = float("inf")

    def _refresh_reward_baselines(self) -> None:
        self.prev_pair_distances = self._pairwise_distances()
        self._refresh_prev_distances(self.prev_pair_distances)
        for j in range(self.num_evaders):
            self.prev_gaps[j] = self._shaping_gap(j)
            self.prev_geometry_scores[j] = self._geometry_score(j)
            members = self.shaping_members(j)
            self.prev_angular_potentials[j] = angular_enclosure_potential(
                self.positions[members],
                self.positions[self.num_pursuers + j],
                self.reward_config.angular_closure_band,
            )
            self.prev_ring_potentials[j] = ring_uniformity_potential(
                self.positions[members],
                self.positions[self.num_pursuers + j],
                self.evader_velocities()[j],
                self.reward_config.ring_radius,
                self.reward_config.ring_lookahead,
            )
            self._shaping_signatures[j] = tuple(self.shaping_members(j))

    # ------------------------------------------------------------ rewards
    def _collision_reward(self, i: int) -> float:
        for k in range(self.max_agents):
            if k == i or not self._agent_active(k):
                continue
            if np.linalg.norm(self.positions[i] - self.positions[k]) < self.collision_distance:
                return self.reward_config.collision_penalty
        return 0.0

    def _boundary_penalty_at(self, pos: np.ndarray) -> float:
        r = self.agent_radius
        if pos[0] < r or pos[0] > self.length - r or pos[1] < r or pos[1] > self.length - r:
            return -10.0
        if pos[0] < r + 0.15 or pos[0] > self.length - r - 0.15 or pos[1] < r + 0.15 or pos[1] > self.length - r - 0.15:
            return -3.0
        if pos[0] < r + 0.35 or pos[0] > self.length - r - 0.35 or pos[1] < r + 0.35 or pos[1] > self.length - r - 0.35:
            return -1.0
        if pos[0] < r + 0.60 or pos[0] > self.length - r - 0.60 or pos[1] < r + 0.60 or pos[1] > self.length - r - 0.60:
            return -0.2
        return 0.0

    def _boundary_reward(self, i: int) -> float:
        return self._boundary_penalty_at(self.positions[i])

    def _predictive_boundary_penalty(self, i: int, cmd: np.ndarray) -> float:
        """IBR-style soft penalty on the predicted next position near fences.

        Mirrors IBRSolver._utility: penalize the one-step lookahead position
        proportionally to how close it gets to the boundary margin.  This lets
        the policy learn to brake *before* crossing the hard boundary.
        """
        margin = self.reward_config.predictive_boundary_margin
        if margin <= 0.0:
            return 0.0
        vel = np.clip(cmd[i], -1.0, 1.0)
        speed = float(np.linalg.norm(vel))
        if speed > 1.0:
            vel = vel / speed
        next_pos = self.positions[i, :2].copy() + vel * self.v_max * self.time_step
        penalty = 0.0
        for dim in range(2):
            near_low = max(0.0, min(1.0, (margin - next_pos[dim]) / margin))
            near_high = max(
                0.0, min(1.0, (next_pos[dim] - (self.length - margin)) / margin)
            )
            penalty += near_low + near_high
        return -self.reward_config.predictive_boundary_weight * penalty

    def _obstacle_reward(self, i: int) -> float:
        for obs in self.obstacles:
            h_dist = np.linalg.norm(self.positions[i, :2] - obs.position)
            if h_dist < obs.radius + self.agent_radius and self.positions[i, 2] < obs.height:
                return self.reward_config.obstacle_penalty
        return 0.0

    def _heading_reward(self, i: int, target: int) -> float:
        v = self.velocities[i]
        speed = float(np.linalg.norm(v))
        if speed < 0.02:
            return 0.0
        to_target = self.positions[self.num_pursuers + target] - self.positions[i]
        d = float(np.linalg.norm(to_target))
        if d <= self.d_cap:
            return 0.0
        return float(np.dot(v / speed, to_target / d))

    def _approach_reward(self, previous: float, current: float) -> float:
        """Reward approach only until the pursuer reaches the capture circle.

        Normalization scale is the per-second closing distance a pursuer can
        achieve at full speed (``v_max`` m/s), so one second of direct pursuit
        is rewarded at magnitude ``distance_weight``; normalizing by the full
        map length (8 m) previously made per-step approach rewards ~30x weaker
        than the time penalty, starving the pursuit incentive.
        """
        if current <= self.d_cap:
            return 0.0
        return self.reward_config.distance_weight * normalized_progress(
            previous, current, self.v_max, self.reward_config.progress_clip
        )

    # ------------------------------------------------------------------ getters
    def snapshot(self) -> Dict:
        return {
            "episode_step": self.episode_step_count,
            "num_pursuers": self.num_pursuers,
            "num_evaders": self.num_evaders,
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "pursuer_mask": self.pursuer_mask.copy(),
            "task_mask": self.task_mask.copy(),
            "safe_mask": self.safe_mask.copy(),
            "stages": self.stages.copy(),
            "captured": self.captured.copy(),
            "roles": self.roles.copy(),
            "prev_roles": self.prev_roles.copy(),
            "assignment": self.assignment.copy(),
            "obstacles": [(o.position, o.radius, o.height) for o in self.obstacles],
            "length": self.length,
            "d_cap": self.d_cap,
            "hold_steps": self.hold_steps,
            "time_step": self.time_step,
            "v_max": self.v_max,
            "v_max_e": self.v_max_e,
        }

    def pursuer_positions(self) -> np.ndarray:
        return self.positions[: self.num_pursuers].copy()

    def evader_positions(self) -> np.ndarray:
        return self.positions[self.num_pursuers : self.num_pursuers + self.num_evaders].copy()

    def pursuer_velocities(self) -> np.ndarray:
        return self.velocities[: self.num_pursuers].copy()

    def evader_velocities(self) -> np.ndarray:
        return self.velocities[self.num_pursuers : self.num_pursuers + self.num_evaders].copy()

    # --------------------------------------------------- partial observability
    @property
    def active_pursuer_count(self) -> int:
        """Number of currently online (masked) pursuers in the episode scale."""
        return int(np.count_nonzero(self.pursuer_mask[: self.num_pursuers]))

    def perceives(self, pursuer_idx: int, entity_idx: int) -> bool:
        """Whether pursuer ``pursuer_idx`` perceives agent ``entity_idx``.

        With ``perception_radius=None`` every entity is perceived (full
        observability, the default).  Otherwise perception is a disk of that
        radius around the pursuer.
        """
        if self.perception_radius is None:
            return True
        return bool(
            np.linalg.norm(self.positions[pursuer_idx] - self.positions[entity_idx])
            <= self.perception_radius
        )

    def pursuer_visibility(self) -> np.ndarray:
        """(num_pursuers, max_agents) bool: which entities each pursuer perceives."""
        vis = np.zeros((self.num_pursuers, self.max_agents), dtype=bool)
        for i in range(self.num_pursuers):
            if not self.pursuer_mask[i]:
                continue
            for k in range(self.max_agents):
                vis[i, k] = self.perceives(i, k)
        return vis

    # ---------------------------------------------- mid-episode online/offline
    def set_pursuer_offline(self, pursuer_idx: int) -> None:
        """A pursuer suddenly drops offline mid-episode (physical removal).

        The pursuer leaves the task graph (excluded from allocation, dynamics,
        collisions and rewards).  The encoder already filters by
        ``pursuer_mask`` and resets its graph hidden state on signature change,
        so coalition reconfiguration happens automatically.
        """
        if not (0 <= pursuer_idx < self.num_pursuers):
            raise ValueError(f"pursuer index {pursuer_idx} out of range")
        self.pursuer_mask[pursuer_idx] = False
        self.roles[pursuer_idx] = ROLE_RES
        self.assignment[pursuer_idx] = -1

    def set_pursuer_online(self, pursuer_idx: int) -> None:
        """A pursuer rejoins mid-episode (its role is reassigned next allocation)."""
        if not (0 <= pursuer_idx < self.num_pursuers):
            raise ValueError(f"pursuer index {pursuer_idx} out of range")
        self.pursuer_mask[pursuer_idx] = True
        self.roles[pursuer_idx] = ROLE_RES
        self.assignment[pursuer_idx] = -1

    def activate_evader(self, evader_idx: int, position: np.ndarray,
                        goal: np.ndarray) -> None:
        """An evader joins the scene mid-episode (previously offline slot).

        The slot must be an inactive padded slot within ``max_evaders``.  The
        new evader enters the task graph in TRK stage with a private goal.
        """
        if not (0 <= evader_idx < self.max_evaders):
            raise ValueError(f"evader index {evader_idx} out of range")
        if self.task_mask[evader_idx]:
            raise ValueError(f"evader slot {evader_idx} is already active")
        self.num_evaders = max(self.num_evaders, evader_idx + 1)
        idx = self.num_pursuers + evader_idx
        self.positions[idx, :2] = np.asarray(position, dtype=np.float64)[:2]
        self.positions[idx, 2] = self.flight_height
        self.velocities[idx] = 0.0
        self.task_mask[evader_idx] = True
        self.safe_mask[evader_idx] = True
        self.stages[evader_idx] = MODE_TRK
        self.captured[evader_idx] = False
        self.hold_counters[evader_idx] = 0
        self.evader_goals[evader_idx, :2] = np.asarray(goal, dtype=np.float64)[:2]
        self.evader_goals[evader_idx, 2] = self.flight_height
        self.prev_gaps[evader_idx] = self._coalition_gap(evader_idx)

    # --------------------------------------------------- legacy baseline compat
    # The comparison baselines (Grouped/Independent/No-Graph MADDPG) and their
    # ST-HGAT/LSTM observation systems expect the historical UAVEnv interface.
    # These aliases make StageCoalitionEnv a drop-in replacement so that every
    # algorithm shares the *same* dynamics, capture judgement, reward and
    # termination logic (fair comparison per the paper protocol).

    @property
    def num_agents(self) -> int:
        return self.num_pursuers + self.num_evaders

    @property
    def multi_current_pos(self) -> List[np.ndarray]:
        return [self.positions[i] for i in range(self.num_agents)]

    @multi_current_pos.setter
    def multi_current_pos(self, value) -> None:
        for i in range(min(len(value), self.max_agents)):
            self.positions[i] = np.asarray(value[i], dtype=np.float64)

    @property
    def multi_current_vel(self) -> List[np.ndarray]:
        return [self.velocities[i] for i in range(self.num_agents)]

    @multi_current_vel.setter
    def multi_current_vel(self, value) -> None:
        for i in range(min(len(value), self.max_agents)):
            self.velocities[i] = np.asarray(value[i], dtype=np.float64)

    @property
    def pursuer_assignment(self) -> List[int]:
        return self.assignment[: self.num_pursuers].tolist()

    @pursuer_assignment.setter
    def pursuer_assignment(self, value) -> None:
        """Legacy setter: every assigned pursuer acts as a core member."""
        for i in range(self.num_pursuers):
            target = int(value[i]) if i < len(value) else -1
            if 0 <= target < self.num_evaders and self.task_mask[target]:
                self.assignment[i] = target
                self.roles[i] = ROLE_CORE
            else:
                self.assignment[i] = -1
                self.roles[i] = ROLE_RES

    @property
    def capture_done(self) -> str:
        return self.outcome

    @property
    def episode_capture_score(self) -> float:
        return float(np.sum(self.captured[: self.num_evaders])) * 0.5

    def assign_targets(self) -> None:
        """Legacy: assign each pursuer to its nearest evader (all core)."""
        for i in range(self.num_pursuers):
            if self.pursuer_mask[i]:
                self.assignment[i] = self._nearest_evader(i)
                self.roles[i] = ROLE_CORE
            else:
                self.assignment[i] = -1
                self.roles[i] = ROLE_RES

    def assign_pursuer_to_nearest(self, pursuer_idx: int) -> None:
        """Legacy baseline compatibility for a member joining online."""
        if not (0 <= pursuer_idx < self.num_pursuers):
            raise ValueError(f"pursuer index {pursuer_idx} out of range")
        if not self.pursuer_mask[pursuer_idx]:
            raise ValueError(f"pursuer {pursuer_idx} is offline")
        self.assignment[pursuer_idx] = self._nearest_evader(pursuer_idx)
        self.roles[pursuer_idx] = ROLE_CORE

    def get_multi_obs(self) -> List[List[float]]:
        """Legacy global observation layout (same as sim_env.get_multi_obs)."""
        total_obs = []
        for i in range(self.num_pursuers):
            pos = self.positions[i]
            vel = self.velocities[i]
            L = self.length
            S_self = [
                pos[0] / L, pos[1] / L, pos[2] / self.world_height,
                vel[0] / self.v_max, vel[1] / self.v_max, vel[2] / self.v_max,
                pos[0] / L, (L - pos[0]) / L, pos[1] / L, (L - pos[1]) / L,
            ]
            S_team = []
            for j in range(self.num_pursuers):
                if j != i:
                    S_team += [self.positions[j][0] / L, self.positions[j][1] / L,
                               self.positions[j][2] / self.world_height]
            S_evaders = []
            for j in range(self.num_evaders):
                e = self.positions[self.num_pursuers + j]
                S_evaders += [e[0] / L, e[1] / L, e[2] / self.world_height]
            assigned = [1.0 if self.assignment[i] == 0 else 0.0,
                        1.0 if self.assignment[i] == 1 else 0.0]
            while len(assigned) < self.num_evaders:
                assigned.append(0.0)
            S_obstacles = self._legacy_obstacle_obs(pos)
            total_obs.append(S_self + S_team + S_evaders + assigned + S_obstacles)
        for i in range(self.num_evaders):
            idx = self.num_pursuers + i
            pos = self.positions[idx]
            vel = self.velocities[idx]
            S_self = [pos[0] / self.length, pos[1] / self.length,
                      vel[0] / self.v_max_e, vel[1] / self.v_max_e]
            S_dist = []
            for j in range(self.num_pursuers):
                d = float(np.linalg.norm(pos - self.positions[j]))
                S_dist.append(d / np.linalg.norm(2 * self.length))
            S_obstacles = self._legacy_obstacle_obs(pos)
            total_obs.append(S_self + S_dist + S_obstacles)
        return total_obs

    def _legacy_obstacle_obs(self, agent_pos: np.ndarray) -> List[float]:
        obs_info = []
        for obs in self.obstacles:
            obs_info.extend([obs.position[0] / self.length, obs.position[1] / self.length])
        return obs_info

    def pursuer_obs_padded(self) -> np.ndarray:
        """Fixed-dimension zero-padded pursuer observations, scale-invariant.

        Returns (max_pursuers, D) with D computed at the MAXIMAL scale
        (max_pursuers pursuers, max_evaders evaders).  Inactive pursuer rows and
        absent teammate/evader slots are zero, so a fixed-dim MLP policy
        (MAPPO/MADDPG) can consume observations of ANY episode scale with one
        network -- variable-scale fairness via zero-padding in state space.

        Per pursuer row (D = 10 + 3*(max_pursuers-1) + 3*max_evaders
        + max_evaders + 2*num_obstacle):
            self(10) | teammates(3 each, padded) | evaders(3 each, padded)
            | assigned one-hot (padded) | obstacles(2 each)
        """
        L = self.length
        D = (10 + 3 * (self.max_pursuers - 1) + 3 * self.max_evaders
             + self.max_evaders + 2 * self.num_obstacle)
        out = np.zeros((self.max_pursuers, D), dtype=np.float32)
        obs_obst = self._legacy_obstacle_obs(np.zeros(3))
        for i in range(self.num_pursuers):
            if not self.pursuer_mask[i]:
                continue
            pos = self.positions[i]
            vel = self.velocities[i]
            row = [
                pos[0] / L, pos[1] / L, pos[2] / self.world_height,
                vel[0] / self.v_max, vel[1] / self.v_max, vel[2] / self.v_max,
                pos[0] / L, (L - pos[0]) / L, pos[1] / L, (L - pos[1]) / L,
            ]
            for k in range(self.max_pursuers - 1):
                if k < self.num_pursuers - 1:
                    t_idx = (i + 1 + k) % self.num_pursuers  # other teammates
                    p = self.positions[t_idx]
                    row += [p[0] / L, p[1] / L, p[2] / self.world_height]
                else:
                    row += [0.0, 0.0, 0.0]
            for j in range(self.max_evaders):
                if j < self.num_evaders and self.task_mask[j]:
                    e = self.positions[self.num_pursuers + j]
                    row += [e[0] / L, e[1] / L, e[2] / self.world_height]
                else:
                    row += [0.0, 0.0, 0.0]
            row += [1.0 if int(self.assignment[i]) == j else 0.0
                    for j in range(self.max_evaders)]
            row += obs_obst
            out[i] = np.asarray(row, dtype=np.float32)
        return out

    def get_grouped_obs(self, pursuer_id: int, teammate_ids: Sequence[int], target_id: int) -> List[float]:
        """3-D grouped observation for the fixed-size MADDPG baselines."""
        pos = self.positions[pursuer_id]
        vel = self.velocities[pursuer_id]
        L = self.length
        S_self = [
            pos[0] / L, pos[1] / L, pos[2] / self.world_height,
            vel[0] / self.v_max, vel[1] / self.v_max, vel[2] / self.v_max,
            pos[0] / L, (L - pos[0]) / L, pos[1] / L, (L - pos[1]) / L,
        ]
        S_teammates = []
        for j in teammate_ids:
            if j != pursuer_id and j < self.num_pursuers:
                p = self.positions[j]
                S_teammates += [p[0] / L, p[1] / L, p[2] / self.world_height]
        while len(S_teammates) < 3 * (self.max_pursuers - 1):
            S_teammates += [0.0, 0.0, 0.0]
        evader_idx = self.num_pursuers + target_id
        t = self.positions[evader_idx]
        S_target = [t[0] / L, t[1] / L, t[2] / self.world_height]
        S_obstacles_near = self._get_nearest_obstacles(pos, k=2)
        return S_self + S_teammates + S_target + S_obstacles_near

    def _get_nearest_obstacles(self, agent_pos: np.ndarray, k: int = 2) -> List[float]:
        ranked = sorted(self.obstacles, key=lambda o: float(np.linalg.norm(np.asarray(agent_pos)[:2] - o.position)))
        obs = []
        for o in ranked[:k]:
            obs += [o.position[0] / self.length, o.position[1] / self.length]
        while len(obs) < 2 * k:
            obs += [0.0, 0.0]
        return obs

    def render(self, *args, **kwargs):  # legacy no-op (headless)
        return None

    def close(self) -> None:
        return None
