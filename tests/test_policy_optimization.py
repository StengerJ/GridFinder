import os
import random
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "examples"
POLICY_ROOT = EXAMPLES_ROOT / "Policy-Optimization"
for path in (ROOT, EXAMPLES_ROOT, POLICY_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import OBSERVATION_CODES, PPOConfig
from generate_map_corpus import generate_map, has_safe_path, write_stage_split
import main as policy_main
from map_corpus import list_stage_map_files
from policy_env import FileBackedGridEnv
from ppo_trainer import PPOTrainer, load_checkpoint_model, run_policy_evaluation


class TestPolicyOptimization(unittest.TestCase):
    """Covers the new file-backed PPO environment, corpus helpers, and smoke training path."""

    def _write_world(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_generated_corpus_counts_exist(self):
        corpus_root = ROOT / "examples" / "maps" / "policy_optimization"
        self.assertEqual(len(list_stage_map_files(corpus_root, "train", 1)), 64)
        self.assertEqual(len(list_stage_map_files(corpus_root, "train", 2)), 64)
        self.assertEqual(len(list_stage_map_files(corpus_root, "train", 3)), 64)
        self.assertEqual(len(list_stage_map_files(corpus_root, "eval", 1)), 16)
        self.assertEqual(len(list_stage_map_files(corpus_root, "eval", 2)), 16)
        self.assertEqual(len(list_stage_map_files(corpus_root, "eval", 3)), 16)

    def test_stage_split_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            left_root = Path(left_dir)
            right_root = Path(right_dir)
            write_stage_split(left_root, "train", 1, 2, 777)
            write_stage_split(right_root, "train", 1, 2, 777)
            left_files = sorted((left_root / "train" / "stage1").glob("*.txt"))
            right_files = sorted((right_root / "train" / "stage1").glob("*.txt"))
            self.assertEqual([path.read_text(encoding="utf-8") for path in left_files], [path.read_text(encoding="utf-8") for path in right_files])

    def test_generate_map_has_safe_path(self):
        world = generate_map(1, random.Random(123))
        grid = [list(row) for row in world.splitlines()]
        start = None
        goal = None
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell == "a":
                    start = (x, y)
                elif cell == "g":
                    goal = (x, y)
        self.assertIsNotNone(start)
        self.assertIsNotNone(goal)
        self.assertTrue(has_safe_path(grid, start, goal))

    def test_environment_wall_collision_goal_visibility_and_frame_shift(self):
        world = "\n".join(
            [
                "wwwww",
                "wa gw",
                "w o w",
                "wwwww",
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            world_path = self._write_world(temp_dir, "env.txt", world)
            env = FileBackedGridEnv([world_path], max_steps=10, seed=0, view_size=5, frame_stack=4)
            try:
                obs = env.reset()
                self.assertEqual(obs.shape, (4, 5, 5))
                self.assertEqual(obs[-1, 2, 4], OBSERVATION_CODES["goal"])
                next_obs, reward, done, info = env.step(3)
                self.assertEqual(reward, -0.01)
                self.assertFalse(done)
                self.assertEqual(info["result"], "ongoing")
                self.assertTrue(np.array_equal(next_obs[:-1], obs[1:]))
            finally:
                env.close()

    def test_environment_goal_hole_timeout_and_sampling(self):
        goal_world = "\n".join(
            [
                "wwwww",
                "wag w",
                "w   w",
                "wwwww",
            ]
        ) + "\n"
        hole_world = "\n".join(
            [
                "wwwww",
                "wa  w",
                "w ogw",
                "wwwww",
            ]
        ) + "\n"
        idle_world = "\n".join(
            [
                "wwwww",
                "wa  w",
                "w  gw",
                "wwwww",
            ]
        ) + "\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            goal_path = self._write_world(temp_dir, "goal.txt", goal_world)
            hole_path = self._write_world(temp_dir, "hole.txt", hole_world)
            idle_path = self._write_world(temp_dir, "idle.txt", idle_world)

            env = FileBackedGridEnv([goal_path], max_steps=10, seed=0, view_size=5, frame_stack=4)
            try:
                env.reset()
                _, reward, done, info = env.step(0)
                self.assertTrue(done)
                self.assertEqual(reward, 1.0)
                self.assertEqual(info["result"], "goal")
            finally:
                env.close()

            env = FileBackedGridEnv([hole_path], max_steps=10, seed=0, view_size=5, frame_stack=4)
            try:
                env.reset()
                env.step(0)
                _, reward, done, info = env.step(1)
                self.assertTrue(done)
                self.assertEqual(reward, -1.0)
                self.assertEqual(info["result"], "hole")
            finally:
                env.close()

            env = FileBackedGridEnv([idle_path], max_steps=1, seed=0, view_size=5, frame_stack=4)
            try:
                env.reset()
                _, reward, done, info = env.step(2)
                self.assertTrue(done)
                self.assertEqual(reward, -0.02)
                self.assertEqual(info["result"], "timeout")
            finally:
                env.close()

            env = FileBackedGridEnv([goal_path, idle_path], max_steps=5, seed=0, view_size=5, frame_stack=4)
            try:
                seen = set()
                for _ in range(20):
                    env.reset()
                    seen.add(env.current_map_path.name)
                self.assertEqual(seen, {"goal.txt", "idle.txt"})
            finally:
                env.close()

    def test_ppo_smoke_training_and_evaluation(self):
        corpus_root = ROOT / "examples" / "maps" / "policy_optimization"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            config = PPOConfig(
                total_steps=32,
                num_envs=2,
                rollout_steps=16,
                ppo_epochs=1,
                num_minibatches=4,
                eval_interval=32,
                eval_episodes=4,
            )
            trainer = PPOTrainer(
                maps_root=corpus_root,
                log_dir=log_dir,
                config=config,
                device="cpu",
                seed=0,
                start_stage=1,
            )
            trainer.learn()

            latest_path = log_dir / "checkpoints" / "latest.pt"
            best_path = log_dir / "checkpoints" / "best.pt"
            self.assertTrue(latest_path.exists())
            self.assertTrue(best_path.exists())
            self.assertTrue((log_dir / "metrics.csv").exists())
            self.assertTrue((log_dir / "config.json").exists())

            device = torch.device("cpu")
            model, _ = load_checkpoint_model(latest_path, device)
            metrics = run_policy_evaluation(
                model,
                maps_root=corpus_root,
                stage=1,
                split="eval",
                episodes=2,
                seed=5,
                device=device,
            )
            self.assertIn("success_rate", metrics)
            self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_main_pipeline_runs_headless(self):
        corpus_root = ROOT / "examples" / "maps" / "policy_optimization"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "main.py",
                    "--maps-root",
                    str(corpus_root),
                    "--log-dir",
                    str(log_dir),
                    "--total-steps",
                    "32",
                    "--num-envs",
                    "2",
                    "--eval-episodes",
                    "2",
                    "--checkpoint",
                    "latest",
                    "--skip-render",
                    "--device",
                    "cpu",
                ]
                exit_code = policy_main.main()
            finally:
                sys.argv = original_argv

            self.assertEqual(exit_code, 0)
            self.assertTrue((log_dir / "checkpoints" / "latest.pt").exists())
