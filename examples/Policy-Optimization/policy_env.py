"""Lightweight grid environment and synchronous vector wrapper for PPO training."""

from collections import deque

import numpy as np
import pygame as pg

from config import ACTION_DELTAS, FRAME_STACK, OBSERVATION_CODES, VIEW_SIZE
from gridworld._compat import Box, Discrete, Env
from gridworld.modules._resources import image_path
from map_corpus import ensure_map_paths


class FileBackedGridEnv(Env):
    """Runs one deterministic partially observable episode over a disk-backed map."""

    def __init__(
        self,
        maps,
        max_steps,
        seed=None,
        view_size=VIEW_SIZE,
        frame_stack=FRAME_STACK,
        cell_size=32,
    ):
        super().__init__()
        if view_size % 2 == 0:
            raise ValueError("view_size must be odd.")

        self.map_bank = ensure_map_paths(maps)
        self.max_steps = max_steps
        self.view_size = view_size
        self.frame_stack = frame_stack
        self.cell_size = cell_size
        self.rng = np.random.default_rng(seed)

        self.action_space = Discrete(len(ACTION_DELTAS))
        self.observation_space = Box(
            low=0,
            high=max(OBSERVATION_CODES.values()),
            shape=(self.frame_stack, self.view_size, self.view_size),
            dtype="int8",
        )

        self._frames = deque(maxlen=self.frame_stack)
        self._current_map = None
        self._grid = ()
        self._agent_position = (0, 0)
        self._step_count = 0
        self._screen = None
        self._surfaces = None

    @property
    def current_map_path(self):
        """Returns the path of the currently loaded map after reset."""

        if self._current_map is None:
            raise RuntimeError("Environment has not been reset.")
        return self._current_map.path

    def _sample_map(self):
        """Samples one map from the loaded map bank with replacement."""

        index = int(self.rng.integers(0, len(self.map_bank)))
        return self.map_bank[index]

    def _symbol_at(self, coord):
        """Returns the raw ASCII symbol at a world coordinate or None outside bounds."""

        x, y = coord
        if y < 0 or y >= len(self._grid) or x < 0 or x >= len(self._grid[0]):
            return None
        return self._grid[y][x]

    def _encode_symbol(self, symbol):
        """Encodes one world symbol into the integer observation vocabulary."""

        if symbol is None:
            return OBSERVATION_CODES["unknown"]
        if symbol == "w":
            return OBSERVATION_CODES["wall"]
        if symbol == "o":
            return OBSERVATION_CODES["hole"]
        if symbol == "g":
            return OBSERVATION_CODES["goal"]
        return OBSERVATION_CODES["empty"]

    def _current_frame(self):
        """Builds the egocentric local observation centered on the agent."""

        radius = self.view_size // 2
        frame = np.full((self.view_size, self.view_size), OBSERVATION_CODES["unknown"], dtype=np.int8)
        agent_x, agent_y = self._agent_position

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                world_coord = (agent_x + dx, agent_y + dy)
                frame[dy + radius, dx + radius] = self._encode_symbol(self._symbol_at(world_coord))

        center_symbol = self._symbol_at(self._agent_position)
        if center_symbol == "g":
            frame[radius, radius] = OBSERVATION_CODES["goal"]
        elif center_symbol == "o":
            frame[radius, radius] = OBSERVATION_CODES["hole"]
        else:
            frame[radius, radius] = OBSERVATION_CODES["agent"]
        return frame

    def _stacked_observation(self):
        """Stacks the frame history into the tensor returned to the policy."""

        return np.stack(tuple(self._frames), axis=0)

    def reset(self):
        """Loads a fresh map, resets episode state, and fills the frame stack."""

        self._current_map = self._sample_map()
        self._grid = self._current_map.rows
        self._agent_position = self._current_map.metadata.starts[0]
        self._step_count = 0

        initial_frame = self._current_frame()
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(initial_frame.copy())
        return self._stacked_observation()

    def step(self, action):
        """Applies one deterministic move and returns the next observation tuple."""

        if self._current_map is None:
            raise RuntimeError("Call reset() before step().")

        dx, dy = ACTION_DELTAS[int(action)]
        candidate = (self._agent_position[0] + dx, self._agent_position[1] + dy)
        symbol = self._symbol_at(candidate)
        if symbol is not None and symbol != "w":
            self._agent_position = candidate

        self._step_count += 1
        current_symbol = self._symbol_at(self._agent_position)
        reward = -0.01
        done = False
        result = "ongoing"

        if current_symbol == "g":
            reward = 1.0
            done = True
            result = "goal"
        elif current_symbol == "o":
            reward = -1.0
            done = True
            result = "hole"
        elif self._step_count >= self.max_steps:
            reward = -0.02
            done = True
            result = "timeout"

        self._frames.append(self._current_frame())
        info = {
            "result": result,
            "success": result == "goal",
            "map_file": str(self.current_map_path),
            "steps": self._step_count,
        }
        return self._stacked_observation(), reward, done, info

    def _load_surfaces(self):
        """Loads and caches the existing gridworld sprites for rendering."""

        if self._surfaces is None:
            size = (self.cell_size, self.cell_size)
            self._surfaces = {
                "wall": pg.transform.scale(pg.image.load(image_path("wall.png")), size),
                "hole": pg.transform.scale(pg.image.load(image_path("hole.png")), size),
                "goal": pg.transform.scale(pg.image.load(image_path("goal.png")), size),
                "agent": pg.transform.scale(pg.image.load(image_path("agent.png")), size),
            }
        return self._surfaces

    def render(self):
        """Draws the current map and agent position with pygame."""

        if self._current_map is None:
            raise RuntimeError("Call reset() before render().")
        if self._screen is None:
            pg.init()
            width = self._current_map.metadata.width * self.cell_size
            height = self._current_map.metadata.height * self.cell_size
            self._screen = pg.display.set_mode((width, height))

        surfaces = self._load_surfaces()
        self._screen.fill((50, 100, 10))
        for row_index, row in enumerate(self._grid):
            for col_index, symbol in enumerate(row):
                pixel = (col_index * self.cell_size, row_index * self.cell_size)
                if symbol == "w":
                    self._screen.blit(surfaces["wall"], pixel)
                elif symbol == "o":
                    self._screen.blit(surfaces["hole"], pixel)
                elif symbol == "g":
                    self._screen.blit(surfaces["goal"], pixel)
        self._screen.blit(
            surfaces["agent"],
            (self._agent_position[0] * self.cell_size, self._agent_position[1] * self.cell_size),
        )
        pg.display.update()
        pg.display.flip()

    def close(self):
        """Closes any pygame resources opened by render()."""

        if self._screen is not None:
            pg.display.quit()
            self._screen = None
        pg.quit()


class SyncVectorEnv:
    """Runs several FileBackedGridEnv instances in-process for synchronous rollouts."""

    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        if not self.envs:
            raise ValueError("SyncVectorEnv requires at least one environment.")
        self.num_envs = len(self.envs)
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self):
        """Resets every managed environment and stacks the observations."""

        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(self, actions):
        """Steps every environment once and auto-resets any finished episodes."""

        next_obs = []
        rewards = []
        dones = []
        infos = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(int(action))
            if done:
                info = dict(info)
                info["terminal_observation"] = obs.copy()
                obs = env.reset()
            next_obs.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(next_obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            infos,
        )

    def close(self):
        """Closes every managed environment."""

        for env in self.envs:
            env.close()
