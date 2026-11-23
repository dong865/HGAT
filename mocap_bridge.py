"""Nokov/TF state reader generalized from pid_swarm_test.py."""

import time

import rclpy
import torch

from crazyflie_examples.fake.swarm.subscriber import TFSubscriber


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)
    return torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )


class Nokov:
    """Return one 28-element state row per discovered Crazyflie."""

    def __init__(self, crazyflies):
        self.log = None
        self.crazyflies = list(crazyflies)
        self.drone_state = torch.zeros((len(self.crazyflies), 28))
        self.drone_state[..., 3] = 1.0
        self.node = TFSubscriber(self._update)
        self.last_time = 0.0
        self.cf_map = {
            cf.prefix.lstrip("/"): index for index, cf in enumerate(self.crazyflies)
        }
        self.seen = torch.zeros(len(self.crazyflies), dtype=torch.bool)
        self.last_wall_update = 0.0

    def _update(self, message):
        self.log = message
        self.last_wall_update = time.monotonic()

    def get_drone_state(self):
        rclpy.spin_once(self.node, timeout_sec=0.02)
        if self.log is None:
            return self.drone_state.clone()
        last_position = self.drone_state[..., :3].clone()
        newest_time = self.last_time
        for transform in self.log.transforms:
            name = transform.child_frame_id.lstrip("/")
            if name not in self.cf_map:
                continue
            index = self.cf_map[name]
            stamp = transform.header.stamp
            newest_time = max(newest_time, stamp.sec + stamp.nanosec / 1e9)
            self.seen[index] = True
            state = self.drone_state[index]
            state[0] = transform.transform.translation.x
            state[1] = transform.transform.translation.y
            state[2] = transform.transform.translation.z
            state[3] = transform.transform.rotation.w
            state[4] = transform.transform.rotation.x
            state[5] = transform.transform.rotation.y
            state[6] = transform.transform.rotation.z
            if state[3] < 0.0:
                state[3:7] *= -1.0
        dt = newest_time - self.last_time
        if dt >= 0.005:
            self.drone_state[..., 7:10] = (
                self.drone_state[..., :3] - last_position
            ) / dt
            self.last_time = newest_time
        return self.drone_state.clone()

    def ready(self) -> bool:
        return bool(torch.all(self.seen))

    def stale(self, timeout: float = 0.25) -> bool:
        return self.last_wall_update == 0.0 or time.monotonic() - self.last_wall_update > timeout

    def close(self) -> None:
        self.node.destroy_node()

