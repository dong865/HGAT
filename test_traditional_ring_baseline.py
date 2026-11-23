import numpy as np

from . import traditional_ring_baseline as ring


class _StaticRingEnv:
    def __init__(self, pursuers, evader=(5.0, 5.0), evader_velocity=(0.0, 0.0)):
        self._pursuers = np.asarray(pursuers, dtype=np.float64)
        self._evaders = np.asarray([evader], dtype=np.float64)
        self._evader_velocities = np.asarray([evader_velocity], dtype=np.float64)
        self.num_evaders = 1
        self.num_pursuers = len(self._pursuers)
        self.length = 20.0
        self.obstacles = []

    def pursuer_positions(self):
        return self._pursuers

    def evader_positions(self):
        return self._evaders

    def evader_velocities(self):
        return self._evader_velocities


def _rotate(points, center, angle):
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return (np.asarray(points) - center) @ rotation.T + center


def test_continuous_ring_closes_radius_and_opens_the_small_angular_gap():
    center = np.array([5.0, 5.0])
    angles = np.array([-0.2, 0.0, np.pi])
    pursuers = center + np.stack([np.cos(angles), np.sin(angles)], axis=1)
    env = _StaticRingEnv(pursuers)
    controller = ring.ContinuousRingGuidanceController(d_cap=0.8, n_p=3)
    controller.assign[:] = 0

    actions = controller.act(env)

    radial = pursuers - center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    tangent = np.stack([-radial[:, 1], radial[:, 0]], axis=1)
    assert np.all(np.sum(actions * radial, axis=1) < 0.0)
    # The member at angle zero has a tiny gap behind and a large gap ahead,
    # so a positive tangential command opens the crowded gap without a slot swap.
    assert float(np.dot(actions[1], tangent[1])) > 0.0


def test_continuous_ring_is_rotation_equivariant_without_obstacles_or_boundaries():
    center = np.array([5.0, 5.0])
    angles = np.array([-0.4, 0.3, 2.7])
    pursuers = center + 1.1 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    controller = ring.ContinuousRingGuidanceController(d_cap=0.8, n_p=3)
    controller.assign[:] = 0
    original = controller.act(_StaticRingEnv(pursuers))

    turn = 0.73
    rotated_positions = _rotate(pursuers, center, turn)
    rotated = controller.act(_StaticRingEnv(rotated_positions))
    expected = _rotate(original, np.zeros(2), turn)

    assert np.allclose(rotated, expected, atol=1e-7)
