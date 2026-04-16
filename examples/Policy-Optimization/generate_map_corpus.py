"""Deterministically generates the committed PPO map corpus from a seed."""

import argparse
import random
from collections import deque
from pathlib import Path

from config import DEFAULT_MAPS_ROOT, STAGE_CONFIGS


def neighbors(coord):
    """Returns the four cardinal neighbors of one cell."""

    x, y = coord
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def has_safe_path(grid, start, goal):
    """Checks whether a start-to-goal path exists without touching walls or holes."""

    frontier = deque([start])
    seen = {start}
    width = len(grid[0])
    height = len(grid)

    while frontier:
        current = frontier.popleft()
        if current == goal:
            return True
        for nx, ny in neighbors(current):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in seen:
                continue
            if grid[ny][nx] in {"w", "o"}:
                continue
            seen.add((nx, ny))
            frontier.append((nx, ny))
    return False


def choose_start_goal(cells, min_distance, rng):
    """Chooses a start and goal pair that satisfy the stage distance threshold."""

    shuffled = list(cells)
    rng.shuffle(shuffled)
    for start in shuffled:
        for goal in shuffled:
            if start == goal:
                continue
            distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if distance >= min_distance:
                return start, goal
    return None


def generate_map(stage_id, rng):
    """Samples one solvable ASCII map for the requested curriculum stage."""

    stage = STAGE_CONFIGS[stage_id]
    while True:
        size = stage.size
        grid = [[" " for _ in range(size)] for _ in range(size)]
        for index in range(size):
            grid[0][index] = "w"
            grid[size - 1][index] = "w"
            grid[index][0] = "w"
            grid[index][size - 1] = "w"

        walkable = []
        for row in range(1, size - 1):
            for col in range(1, size - 1):
                roll = rng.random()
                if roll < stage.wall_probability:
                    grid[row][col] = "w"
                elif roll < stage.wall_probability + stage.hole_probability:
                    grid[row][col] = "o"
                else:
                    walkable.append((col, row))

        if len(walkable) < 2:
            continue

        pair = choose_start_goal(walkable, stage.min_goal_distance, rng)
        if pair is None:
            continue

        start, goal = pair
        grid[start[1]][start[0]] = "a"
        grid[goal[1]][goal[0]] = "g"
        if has_safe_path(grid, start, goal):
            return "\n".join("".join(row) for row in grid)


def write_stage_split(output_root, split, stage_id, count, base_seed):
    """Writes one full stage split to disk using deterministic per-file seeds."""

    stage_dir = output_root / split / f"stage{stage_id}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        rng = random.Random(base_seed + stage_id * 10_000 + index)
        world_text = generate_map(stage_id, rng)
        (stage_dir / f"map_{index:03d}.txt").write_text(world_text + "\n", encoding="utf-8")


def clear_output_root(output_root):
    """Removes previously generated files while keeping the root directory in place."""

    if not output_root.exists():
        return
    for path in sorted(output_root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def main():
    """CLI entry point for generating the full PPO map corpus."""

    parser = argparse.ArgumentParser(description="Generate the PPO map corpus.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force:
        clear_output_root(args.output_root)

    for stage_id, stage in STAGE_CONFIGS.items():
        write_stage_split(args.output_root, "train", stage_id, stage.train_count, args.seed)
        write_stage_split(args.output_root, "eval", stage_id, stage.eval_count, args.seed + 1_000_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
