"""Shared deterministic scenes and private-goal evader dynamics."""

from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

PROTOCOL_VERSION = "goal_directed_v2_goal_margin"


class ScenarioGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScenarioConfig:
    flight_height: float = 1.0
    length: float = 8.0
    num_obstacles: int = 10
    agent_radius: float = 0.2
    collision_distance: float = 0.4
    obstacle_radius_min: float = 0.12
    obstacle_radius_max: float = 0.22
    obstacle_height: float = 3.0
    obstacle_gap: float = 0.35
    entity_clearance: float = 0.65
    boundary_margin: float = 0.7
    goal_boundary_margin: float = 1.6
    min_agent_separation: float = 0.55
    min_pursuer_evader: float = 1.5
    min_evader_separation: float = 1.2
    min_goal_separation: float = 1.2
    min_goal_distance: float = 3.0
    max_goal_distance: float = 5.5
    goal_radius: float = 0.25
    capture_radius: float = 2.0  # protocol d_cap (DEFAULT_D_CAP)
    grid_resolution: float = 0.1
    max_attempts: int = 500
    max_speed: float = 0.35
    goal_weight: float = 3.0
    pursuer_weight: float = 0.3
    obstacle_weight: float = 1.8
    boundary_weight: float = 0.8
    sibling_weight: float = 0.25
    tangent_weight: float = 0.35
    pursuer_influence: float = 2.5
    max_pursuer_repulsion: float = 2.0
    obstacle_influence: float = 0.9


@dataclass(frozen=True)
class ScenarioState:
    pursuer_positions: np.ndarray  # (N, 3) [x, y, z]
    evader_positions: np.ndarray  # (N, 3) [x, y, z]
    evader_goals: np.ndarray  # (N, 3) [x, y, z]
    obstacles: np.ndarray  # [x, y, radius, height]
    seed: int
    protocol_version: str = PROTOCOL_VERSION


def derive_scenario_seed(root_seed: int, split: str, episode: int, scale: Tuple[int, int]) -> int:
    split_ids = {"train": 11, "validation": 23, "test": 37}
    if split not in split_ids:
        raise ValueError(f"unknown seed split {split!r}")
    p, e = int(scale[0]), int(scale[1])
    value = (int(root_seed) * 1000003 + split_ids[split] * 1000033 + int(episode) * 9176 + p * 131 + e * 17)
    return int(value % (2**32 - 1))


def _angular_gap(points: np.ndarray, center: np.ndarray) -> float:
    if len(points) < 3:
        return float("inf")
    angles = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    return float(np.max(gaps))


def _path_exists(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, cfg: ScenarioConfig) -> bool:
    n = int(np.ceil(cfg.length / cfg.grid_resolution)) + 1
    occupied = np.zeros((n, n), dtype=bool)
    coords = np.arange(n, dtype=np.float64) * cfg.grid_resolution
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    for x, y, radius, *_ in obstacles:
        occupied |= (xx - x) ** 2 + (yy - y) ** 2 <= (radius + cfg.agent_radius + 0.08) ** 2
    margin_cells = int(np.ceil(cfg.agent_radius / cfg.grid_resolution))
    occupied[:margin_cells] = True
    occupied[-margin_cells:] = True
    occupied[:, :margin_cells] = True
    occupied[:, -margin_cells:] = True
    s = tuple(np.clip(np.rint(start / cfg.grid_resolution).astype(int), 0, n - 1))
    g = tuple(np.clip(np.rint(goal / cfg.grid_resolution).astype(int), 0, n - 1))
    if occupied[s] or occupied[g]:
        return False
    queue = deque([s])
    visited = {s}
    while queue:
        cell = queue.popleft()
        if cell == g:
            return True
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + delta[0], cell[1] + delta[1])
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in visited and not occupied[nxt]:
                visited.add(nxt)
                queue.append(nxt)
    return False


def validate_scenario(state: ScenarioState, cfg: ScenarioConfig) -> Dict[str, object]:
    positions = np.vstack((state.pursuer_positions, state.evader_positions))
    errors = []
    if len(state.obstacles) != cfg.num_obstacles:
        errors.append("obstacle_count")
    for i in range(len(positions)):
        if np.any(positions[i, :2] < cfg.boundary_margin) or np.any(positions[i, :2] > cfg.length - cfg.boundary_margin):
            errors.append("entity_boundary")
        for k in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i, :2] - positions[k, :2]) < cfg.min_agent_separation:
                errors.append("entity_separation")
    for goal in state.evader_goals:
        if np.any(goal[:2] < cfg.goal_boundary_margin) or np.any(goal[:2] > cfg.length - cfg.goal_boundary_margin):
            errors.append("goal_boundary")
    for i, evader in enumerate(state.evader_positions):
        distance = np.linalg.norm(state.evader_goals[i, :2] - evader[:2])
        if not cfg.min_goal_distance <= distance <= cfg.max_goal_distance:
            errors.append("goal_distance")
        if not _path_exists(evader[:2], state.evader_goals[i, :2], state.obstacles, cfg):
            errors.append("unreachable_goal")
        # initial-capture rejection must use the *protocol* capture radius
        # (was a stale hardcoded 1.2 before the protocol moved to d_cap=2.0)
        d_cap_protocol = cfg.capture_radius
        close = state.pursuer_positions[
            np.linalg.norm(state.pursuer_positions[:, :2] - evader[:2], axis=1) <= d_cap_protocol
        ]
        if len(close) >= 3 and _angular_gap(close[:, :2], evader[:2]) < np.pi:
            errors.append("initial_capture")
    for point in np.vstack((positions, state.evader_goals)):
        for x, y, radius, *_ in state.obstacles:
            if np.linalg.norm(point[:2] - np.array([x, y])) < radius + cfg.entity_clearance:
                errors.append("obstacle_clearance")
    return {"valid": not errors, "errors": tuple(sorted(set(errors)))}


class GoalDirectedScenarioSampler:
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or ScenarioConfig()

    def sample(self, n_pursuers: int, n_evaders: int, seed: int) -> ScenarioState:
        cfg = self.config
        for attempt in range(cfg.max_attempts):
            rng = np.random.default_rng(int(seed) + attempt * 104729)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            normal = np.array([-direction[1], direction[0]])
            center = np.full(2, cfg.length / 2.0)
            travel = rng.uniform(cfg.min_goal_distance + 0.2, min(cfg.max_goal_distance - 0.2, 4.8))
            starts, goals = [], []
            z = cfg.flight_height
            for j in range(n_evaders):
                lane = (j - (n_evaders - 1) / 2.0) * 1.25 + rng.uniform(-0.18, 0.18)
                start_xy = center - direction * travel / 2.0 + normal * lane + rng.normal(0.0, 0.12, 2)
                goal_xy = center + direction * travel / 2.0 + normal * lane + rng.normal(0.0, 0.12, 2)
                start_xy = np.clip(start_xy, cfg.boundary_margin, cfg.length - cfg.boundary_margin)
                goal_xy = np.clip(goal_xy, cfg.goal_boundary_margin, cfg.length - cfg.goal_boundary_margin)
                starts.append(np.array([start_xy[0], start_xy[1], z]))
                goals.append(np.array([goal_xy[0], goal_xy[1], z]))
            evaders = np.asarray(starts)
            goals_arr = np.asarray(goals)
            if n_evaders > 1:
                if min(np.linalg.norm(evaders[i, :2] - evaders[k, :2]) for i in range(n_evaders) for k in range(i + 1, n_evaders)) < cfg.min_evader_separation:
                    continue
                if min(np.linalg.norm(goals_arr[i, :2] - goals_arr[k, :2]) for i in range(n_evaders) for k in range(i + 1, n_evaders)) < cfg.min_goal_separation:
                    continue
            pursuers = []
            for i in range(n_pursuers):
                target = evaders[i % n_evaders]
                placed = False
                for _ in range(200):
                    angle = rng.uniform(0.0, 2.0 * np.pi)
                    radius = rng.uniform(1.8, 3.0)
                    pos_xy = target[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
                    if np.any(pos_xy < cfg.boundary_margin) or np.any(pos_xy > cfg.length - cfg.boundary_margin):
                        continue
                    if any(np.linalg.norm(pos_xy - p[:2]) < cfg.min_agent_separation for p in pursuers):
                        continue
                    if np.min(np.linalg.norm(evaders[:, :2] - pos_xy, axis=1)) < cfg.min_pursuer_evader:
                        continue
                    pursuers.append(np.array([pos_xy[0], pos_xy[1], z]))
                    placed = True
                    break
                if not placed:
                    break
            if len(pursuers) != n_pursuers:
                continue
            protected = np.vstack((np.asarray(pursuers)[:, :2], evaders[:, :2], goals_arr[:, :2]))
            obstacles = []
            for _ in range(cfg.num_obstacles):
                placed = False
                for _ in range(300):
                    radius = rng.uniform(cfg.obstacle_radius_min, cfg.obstacle_radius_max)
                    pos = rng.uniform(cfg.boundary_margin, cfg.length - cfg.boundary_margin, 2)
                    if np.min(np.linalg.norm(protected - pos, axis=1)) < radius + cfg.entity_clearance:
                        continue
                    if any(np.linalg.norm(pos - o[:2]) < radius + o[2] + cfg.obstacle_gap for o in obstacles):
                        continue
                    obstacles.append(np.array([pos[0], pos[1], radius, cfg.obstacle_height]))
                    placed = True
                    break
                if not placed:
                    break
            if len(obstacles) != cfg.num_obstacles:
                continue
            state = ScenarioState(np.asarray(pursuers), evaders, goals_arr, np.asarray(obstacles), int(seed))
            if validate_scenario(state, cfg)["valid"]:
                return state
        raise ScenarioGenerationError(f"failed to generate a valid {n_pursuers}v{n_evaders} scene for seed {seed}")


def goal_reached(position: np.ndarray, goal: np.ndarray, radius: float = 0.25) -> bool:
    return bool(np.linalg.norm(np.asarray(position) - np.asarray(goal)) <= radius)


class GoalDirectedPotentialField:
    def __init__(self, config: Optional[ScenarioConfig] = None):
        self.config = config or ScenarioConfig()

    def velocity(self, evader_index: int, pursuer_positions: np.ndarray, evader_positions: np.ndarray,
                 goals: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        cfg = self.config
        pos_full = np.asarray(evader_positions[evader_index], dtype=np.float64)
        pos = pos_full[:2]  # evader moves in the horizontal plane
        goal_pos = np.asarray(goals[evader_index], dtype=np.float64)[:2]
        goal_delta = goal_pos - pos
        goal_dist = float(np.linalg.norm(goal_delta))
        goal_dir = goal_delta / max(goal_dist, 1e-9)
        force = cfg.goal_weight * np.tanh(goal_dist / 1.5) * goal_dir
        for pursuer in np.asarray(pursuer_positions):
            diff = pos - pursuer[:2]
            dist = float(np.linalg.norm(diff))
            if 1e-6 < dist < cfg.pursuer_influence:
                rep = cfg.pursuer_weight * (1.0 / dist - 1.0 / cfg.pursuer_influence) * diff / (dist * dist)
                rep_norm = float(np.linalg.norm(rep))
                if rep_norm > cfg.max_pursuer_repulsion:
                    rep = rep * (cfg.max_pursuer_repulsion / rep_norm)
                force += rep
        obstacle_force = np.zeros(2)
        for x, y, radius, *_ in np.asarray(obstacles):
            diff = pos - np.array([x, y])
            dist = float(np.linalg.norm(diff))
            safe = radius + cfg.agent_radius + cfg.obstacle_influence
            if 1e-6 < dist < safe:
                obstacle_force += cfg.obstacle_weight * (1.0 / dist - 1.0 / safe) * diff / (dist * dist)
        force += obstacle_force
        margin = 0.8
        for axis in range(2):
            if pos[axis] < margin:
                force[axis] += cfg.boundary_weight * (margin - pos[axis]) / margin
            elif pos[axis] > cfg.length - margin:
                force[axis] -= cfg.boundary_weight * (pos[axis] - (cfg.length - margin)) / margin
        for other, other_pos in enumerate(np.asarray(evader_positions)):
            if other == evader_index:
                continue
            diff = pos - other_pos[:2]
            dist = float(np.linalg.norm(diff))
            if 1e-6 < dist < 1.2:
                force += cfg.sibling_weight * (1.0 / dist - 1.0 / 1.2) * diff / (dist * dist)
        if np.linalg.norm(obstacle_force) > 0.4 and np.dot(force, goal_dir) < 0.15:
            tangent = np.array([-obstacle_force[1], obstacle_force[0]])
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            if np.dot(tangent, goal_dir) < 0.0:
                tangent = -tangent
            force += cfg.tangent_weight * tangent
        magnitude = float(np.linalg.norm(force))
        if magnitude < 1e-9:
            return np.zeros(2)
        return force / magnitude * min(cfg.max_speed, magnitude * cfg.max_speed)
