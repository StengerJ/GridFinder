"""Command-line entry point for deterministic A* runs on disk-backed search maps."""

import argparse
from pathlib import Path
import time

import pygame as pg

try:
    from . import path_setup
except ImportError:
    import path_setup

from examples.search.Astar.astar import aStarSearch
from gridworld import GridWorld
from utilities.map_loader import load_world


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"
MAP_ROOT = EXAMPLES_ROOT / "maps" / "search"

WORLD_CONFIG = {
    "small": {
        "dir": MAP_ROOT / "small",
        "kwargs": {"max_episode_step": 1000, "random_state": 42},
    },
    "big": {
        "dir": MAP_ROOT / "large",
        "kwargs": {
            "max_episode_step": 2000,
            "blocksize": (17, 17),
            "random_state": 42,
        },
    },
}


def get_world_path(world_name: str, variant: int) -> Path:
    """Resolves a 1-based world variant to its text-file path."""

    if variant < 1:
        raise ValueError("--variant must be at least 1.")
    config = WORLD_CONFIG[world_name]
    world_path = config["dir"] / f"map_{variant:03d}.txt"
    if not world_path.exists():
        raise FileNotFoundError(f"Map variant not found: {world_path}")
    return world_path


def build_env(world_name: str, variant: int = 1) -> GridWorld:
    """Builds a deterministic GridWorld from a selected disk-backed map variant."""

    config = WORLD_CONFIG[world_name]
    world_text, _ = load_world(get_world_path(world_name, variant), min_starts=1, max_starts=1, min_goals=1)
    return GridWorld(world_text, slip=0.0, log=False, **config["kwargs"])


def format_report(world_name: str, variant: int, result, env: GridWorld) -> str:
    """Formats the search result into the short report written to disk and stdout."""

    lines = [
        f"world={world_name}",
        f"variant={variant}",
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
    """Replays the discovered A* path in the pygame window."""

    step_delay = 0.12 / speed
    env.reset()
    env.render()
    for action in actions:
        env.step(action, testing=True)
        env.render()
        time.sleep(step_delay)


def wait_for_close(env: GridWorld) -> None:
    """Keeps the final rendered world visible until the user closes it."""

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
    """Parses CLI args, runs A*, writes the report, and optionally replays the path."""

    parser = argparse.ArgumentParser(description="Run deterministic A* search on a GridWorld.")
    parser.add_argument("--world", choices=sorted(WORLD_CONFIG), default="small")
    parser.add_argument("--variant", type=int, default=1, help="1-based map variant index within the selected world set.")
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

    env = build_env(args.world, args.variant)
    try:
        result = aStarSearch(env)
        report = format_report(args.world, args.variant, result, env)

        logs_dir = REPO_ROOT / "logs" / "astar"
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_path = logs_dir / f"{args.world}_{args.variant:03d}_report.txt"
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
