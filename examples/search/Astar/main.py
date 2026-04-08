from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import pygame as pg

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from examples.search.Astar.astar import aStarSearch
from gridworld import GridWorld
from library.gridenv import big_world, small_world



WORLD_CONFIG = {
    "small": {
        "world": small_world,
        "kwargs": {"max_episode_step": 1000, "random_state": 42},
    },
    "big": {
        "world": big_world,
        "kwargs": {
            "max_episode_step": 2000,
            "blocksize": (17, 17),
            "random_state": 42,
        },
    },
}


def build_env(world_name: str) -> GridWorld:
    config = WORLD_CONFIG[world_name]
    return GridWorld(config["world"], slip=0.0, log=False, **config["kwargs"])


def format_report(world_name: str, result, env: GridWorld) -> str:
    lines = [
        f"world={world_name}",
        f"start={(int(env.agent.initial_position.x), int(env.agent.initial_position.y))}",
        f"found={result.found}",
        f"goal={result.goal}",
        f"path_length={len(result.path)}",
        f"step_cost={result.step_cost}",
        f"expanded_nodes={result.expanded_nodes}",
        f"actions={[env.action_map[action] for action in result.actions]}",
        f"path={result.path}",
    ]
    return "\n".join(lines) + "\n"


def replay(env: GridWorld, actions: list[int], speed: float) -> None:
    step_delay = 0.12 / speed
    env.reset()
    env.render()
    for action in actions:
        env.step(action, testing=True)
        env.render()
        time.sleep(step_delay)


def wait_for_close(env: GridWorld) -> None:
    while True:
        should_close = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                should_close = True
            elif event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_RETURN, pg.K_SPACE):
                should_close = True
        if should_close:
            return
        env.render()
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic A* search on a GridWorld.")
    parser.add_argument("--world", choices=sorted(WORLD_CONFIG), default="small")
    parser.add_argument("--no-render", action="store_true", help="Skip the graphical replay.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Animation speed multiplier. Higher is faster, lower is slower.",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0.")

    env = build_env(args.world)
    try:
        result = aStarSearch(env)
        report = format_report(args.world, result, env)

        logs_dir = REPO_ROOT / "logs" / "astar"
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_path = logs_dir / f"{args.world}_report.txt"
        report_path.write_text(report, encoding="utf-8")

        print(report, end="")
        if result.found and not args.no_render:
            replay(env, result.actions, args.speed)
            wait_for_close(env)
        return 0 if result.found else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
