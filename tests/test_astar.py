import os
from pathlib import Path
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from examples.search.Astar.astar import aStarSearch, iterateThroughNeighbors, h
from examples.search.Astar.main import build_env, get_world_path
from gridworld import GridWorld


TEST_WORLD = """
wwwww
wa  w
w wgw
w   w
wwwww
"""


NO_PATH_WORLD = """
wwwww
wawgw
wwwww
"""


class TestAstar(unittest.TestCase):
    """Exercises both the search logic and the disk-backed world selection."""

    def test_manhattan_uses_nearest_goal(self):
        self.assertEqual(h((1, 1), [(4, 1), (2, 5)]), 3)

    def test_neighbors_skip_walls(self):
        env = GridWorld(TEST_WORLD, slip=0.0, random_state=0)
        try:
            neighbors = list(iterateThroughNeighbors(env, (1, 1)))
            self.assertEqual(sorted(neighbors), [(0, (2, 1)), (1, (1, 2))])
        finally:
            env.close()

    def test_astar_returns_legal_path(self):
        env = GridWorld(TEST_WORLD, slip=0.0, random_state=0)
        try:
            result = aStarSearch(env)
            self.assertTrue(result.found)
            self.assertEqual(result.path[0], (1, 1))
            self.assertEqual(result.goal, (3, 2))
            self.assertEqual(result.step_cost, len(result.actions))
            self.assertEqual(len(result.path), len(result.actions) + 1)

            current = result.path[0]
            for action, next_coord in zip(result.actions, result.path[1:]):
                valid_moves = dict(iterateThroughNeighbors(env, current))
                self.assertEqual(valid_moves[action], next_coord)
                current = next_coord
        finally:
            env.close()

    def test_astar_reports_no_path(self):
        env = GridWorld(NO_PATH_WORLD, slip=0.0, random_state=0)
        try:
            result = aStarSearch(env)
            self.assertFalse(result.found)
            self.assertEqual(result.path, [])
            self.assertEqual(result.actions, [])
        finally:
            env.close()

    def test_disk_backed_small_variant_builds(self):
        env = build_env("small", 1)
        try:
            result = aStarSearch(env)
            self.assertTrue(result.found)
        finally:
            env.close()

    def test_world_variant_path_exists_on_disk(self):
        path = get_world_path("big", 25)
        self.assertIsInstance(path, Path)
        self.assertTrue(path.exists())
