from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"
for path in (ROOT, EXAMPLES_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from examples.library.dqnalgo import dqn
from examples.library.mpe import SubprocVecEnv
from examples.library.npgalgo import NPG
from examples.library.nn import ActorCritic
from examples.library.ppoalgo import PPO2
from examples.library.trpoalgo import TRPO
from gridworld._compat import Box, Discrete


class DummyEnvContainer:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs


class DummyTestEnv:
    def reset(self):
        return np.zeros(4, dtype=np.float32)

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, True, {}

    def close(self):
        return None


class CounterEnv:
    def __init__(self):
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=10, shape=(1,), dtype="int32")
        self.value = 0

    def reset(self):
        self.value = 0
        return np.array([self.value], dtype=np.int32)

    def step(self, action):
        self.value += int(action) + 1
        done = self.value >= 2
        return np.array([self.value], dtype=np.int32), float(self.value), done, {}

    def close(self):
        return None


def make_counter_env():
    return CounterEnv()


class TestRLSmoke(unittest.TestCase):
    def test_dqn_learns_one_batch(self):
        agent = dqn(input_size=4, action_size=2, memory_size=8, exp_name="TEST", seed=0)
        for _ in range(8):
            state = np.zeros(4, dtype=np.float32)
            next_state = np.ones(4, dtype=np.float32)
            agent.memory.push(state, 0, 1.0, next_state, 1)
        loss = agent.learn(batch_size=4)
        self.assertGreaterEqual(loss, 0.0)

    def test_actor_critic_algorithms_train_one_batch(self):
        states = np.random.random((8, 4)).astype(np.float32)
        actions = np.random.randint(0, 2, size=(8,), dtype=np.int32)

        def make_batch(actor_critic):
            old_log_probs, old_values, _ = actor_critic.log_prob_value_entropy(states, actions)
            returns = old_values + 1.0
            advantage = returns - old_values
            return old_log_probs, old_values, returns, advantage

        ppo_ac = ActorCritic(input_size=4, action_size=2, seed=0)
        old_log_probs, old_values, returns, advantage = make_batch(ppo_ac)
        ppo = PPO2(DummyEnvContainer(1), DummyTestEnv(), ppo_ac, only_test=True)
        pi_loss, v_loss, ent_loss = ppo._train(states, actions, old_log_probs, advantage, old_values, returns)
        self.assertTrue(np.isfinite([pi_loss, v_loss, ent_loss]).all())

        trpo_ac = ActorCritic(input_size=4, action_size=2, seed=1)
        old_log_probs, old_values, returns, advantage = make_batch(trpo_ac)
        trpo = TRPO(DummyEnvContainer(1), DummyTestEnv(), trpo_ac, only_test=True)
        surr_loss, v_loss = trpo._train(states, actions, old_log_probs, advantage, old_values, returns)
        self.assertTrue(np.isfinite([surr_loss, v_loss]).all())

        npg_ac = ActorCritic(input_size=4, action_size=2, seed=2)
        old_log_probs, old_values, returns, advantage = make_batch(npg_ac)
        npg = NPG(DummyEnvContainer(1), DummyTestEnv(), npg_ac, only_test=True)
        surr_loss, v_loss = npg._train(states, actions, old_log_probs, advantage, old_values, returns)
        self.assertTrue(np.isfinite([surr_loss, v_loss]).all())

    def test_subproc_vec_env_round_trip(self):
        envs = SubprocVecEnv([make_counter_env, make_counter_env])
        try:
            obs = envs.reset()
            self.assertEqual(obs.shape, (2, 1))
            next_obs, rewards, dones, infos = envs.step(np.array([0, 1]))
            self.assertEqual(next_obs.shape, (2, 1))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(dones.shape, (2,))
            self.assertEqual(len(infos), 2)
        finally:
            envs.close()
