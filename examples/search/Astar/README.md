# A* Search Example

This example runs deterministic A* search on the GridWorld layouts using the Manhattan distance heuristic.

## Behavior

- Movement is deterministic with `slip=0.0`.
- The search expands 4-neighbor moves only.
- Walls are blocked because they are absent from `env.state_dict`.
- Hole states are excluded from expansion.
- If multiple goal cells exist, the heuristic uses the minimum Manhattan distance to any goal cell.

## Usage

```bash
python examples/search/Astar/main.py --world small
python examples/search/Astar/main.py --world big
python examples/search/Astar/main.py --world small --no-render
python examples/search/Astar/main.py --world small --speed 0.5
python examples/search/Astar/main.py --world small --speed 2.0
```

By default the script opens the gridworld window, animates the A* path step by step, and keeps the solved world visible until you close the window or press `Esc`, `Enter`, or `Space`.
Use `--speed` to control the animation rate: values above `1.0` are faster, and values below `1.0` are slower.

Each run also prints a short report and writes it to `logs/astar/<world>_report.txt`.
