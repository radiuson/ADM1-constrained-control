"""Restrict the environment to the feed rate alone.

The conventional controllers regulate the feed rate and leave the feed
strength at whatever the scenario supplies; a plant that cannot change its
feed composition is in that position.  Comparing a two-input policy against
such a baseline gives the policy an actuator the baseline does not have.

This wrapper removes it.  The policy sets only the first action and the
second is held at a fixed value.  Both are expressed in the units of the
wrapped environment, so the wrapper works whether or not the actions have
already been normalised: `hold` is the value passed through for the second
component, which for a normalised environment is the point of its own range
that corresponds to the scenario's strength.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class FeedRateOnly(gym.Wrapper):
    def __init__(self, env, hold=None, mult=None):
        super().__init__(env)
        lo = np.asarray(env.action_space.low, dtype=np.float64)
        hi = np.asarray(env.action_space.high, dtype=np.float64)
        if lo.shape[0] != 2:
            raise ValueError(f'expected a two-component action, got {lo.shape}')

        if hold is not None:
            self.hold = float(hold)
        elif mult is not None:
            # A multiplier in physical units; map it onto the wrapped range.
            # An unnormalised env has that range directly; a normalised one
            # carries [-1, 1], and the physical bounds are 0.7 to 1.3.
            if abs(lo[1] + 1.0) < 1e-9 and abs(hi[1] - 1.0) < 1e-9:
                self.hold = 2.0 * (float(mult) - 0.7) / (1.3 - 0.7) - 1.0
            else:
                self.hold = float(mult)
        else:
            self.hold = float(lo[1] + 0.5 * (hi[1] - lo[1]))

        self.action_space = gym.spaces.Box(
            low=np.array([lo[0]], dtype=np.float32),
            high=np.array([hi[0]], dtype=np.float32),
            dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        return self.env.step(np.array([a[0], self.hold], dtype=np.float32))
