# GridWorld

<img src="./assets/gridworldenv.png" align="left" width="40%"/>GridWorld is a small grid-environment project for experimenting with pathfinding and partially observable policy learning. The repository is now intentionally focused on two search paths only:

- deterministic A* search
- the file-backed PyTorch PPO example

The core `gridworld` package still provides the environment, rendering, transition model construction, and the gym-like `reset` / `step` interface used by those two paths.
<br clear="left"/>

# Installation

This repo targets Python 3.13+ on CPython and uses `uv`.

```bash
git clone https://github.com/StengerJ/GridFinder
cd GridFinder
uv python install 3.13
uv venv .venv --python 3.13
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r Requirements.txt
uv pip install -e .
```

# Core Grid Usage

```python
from gridworld import GridWorld

world = """
wwwww
wa gw
w o w
wwwww
"""

env = GridWorld(world, slip=0.0, random_state=7)
state = env.reset()
next_state, reward, done, info = env.step(0, testing=True)

print(state, next_state, reward, done, info)
print(env.P_sas.shape, env.R_sa.shape)
```

# A* Search

The deterministic A* example lives in `examples/search/Astar/` and reads committed maps from `examples/maps/search/`.

- `25` small maps are stored in `examples/maps/search/small/`
- `25` large maps are stored in `examples/maps/search/large/`

Example commands:

```bash
python examples/search/Astar/main.py --world small
python examples/search/Astar/main.py --world big
python examples/search/Astar/main.py --world small --variant 7
python examples/search/Astar/main.py --world big --variant 12 --no-render
```

# File-Backed PPO

The PPO example lives in `examples/Policy-Optimization/` and trains on committed map corpora stored under `examples/maps/policy_optimization/`.

- `train/stage1`, `train/stage2`, `train/stage3` each contain `64` maps
- `eval/stage1`, `eval/stage2`, `eval/stage3` each contain `16` held-out maps

Useful commands:

```bash
python examples/Policy-Optimization/generate_map_corpus.py --force
python examples/Policy-Optimization/train_ppo.py
python examples/Policy-Optimization/evaluate_policy.py --checkpoint logs/policy_optimization/checkpoints/best.pt --stage 3
python examples/Policy-Optimization/render_episode.py --checkpoint logs/policy_optimization/checkpoints/best.pt --stage 3
```

More detail is in [examples/Policy-Optimization/README.md](examples/Policy-Optimization/README.md).

# Tests

Run the current test suite with:

```bash
python -m unittest discover -s tests
```
