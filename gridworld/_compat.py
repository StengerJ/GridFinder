from __future__ import annotations

import numpy as np


class Env:
    """Small subset of the gym.Env API used by this project."""

    action_space: object
    observation_space: object

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        return None


class Discrete:
    def __init__(self, n: int):
        self.n = n
        if self.n <= 0:
            raise ValueError("Discrete space size must be positive.")

    def sample(self) -> int:
        return int(np.random.randint(self.n))


class Box:
    def __init__(self, low, high, shape, dtype="float32"):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)

    def sample(self):
        if np.issubdtype(self.dtype, np.integer):
            high = self.high.astype(np.int64) + 1
            return np.random.randint(self.low, high, size=self.shape, dtype=self.dtype)
        return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
