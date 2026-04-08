from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from examples.library.gridenv import small_env_fn
from gridworld import GridWorld, ractGridWorld


WORLD = """
wwwww
wa gw
w o w
wwwww
"""


class TestGridWorld(unittest.TestCase):
    def test_small_env_fn_uses_random_state_keyword(self):
        env = small_env_fn(7)
        try:
            self.assertIsNotNone(env.reset())
        finally:
            env.close()

    def test_gridworld_builds_model_and_steps(self):
        env = GridWorld(WORLD, slip=0.0, random_state=7)
        try:
            state = env.reset()
            self.assertEqual(state, 0)
            next_state, reward, done, info = env.step(0, testing=True)
            self.assertEqual(next_state, 1)
            self.assertEqual(reward, -1)
            self.assertFalse(done)
            self.assertEqual(info, {})
            self.assertEqual(env.P_sas.shape, (env.state_count, env.action_size, env.state_count))
            self.assertEqual(env.R_sa.shape, (env.state_count, env.action_size))
            self.assertTrue(np.allclose(env.P_sas.sum(axis=2), 1.0))
        finally:
            env.close()

    def test_repeated_action_wrapper_shapes_state(self):
        env = ractGridWorld(WORLD, slip=0.0, random_state=7, isDRL=True, viewsize=1, repeat_act=2)
        try:
            state = env.reset()
            self.assertEqual(state.shape, env.observation_space.shape)
        finally:
            env.close()
