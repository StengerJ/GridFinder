# File-Backed Hidden-Goal PPO

This example trains a PPO agent from scratch with PyTorch on a partially observable grid task.
The training and evaluation maps live on disk under `examples/maps/policy_optimization/`, and each episode samples a committed text file instead of regenerating a map at reset time.

## Map Layout

- `examples/maps/search/` stores the shared A* worlds.
- `examples/maps/policy_optimization/train/stage{1,2,3}/` stores 64 training maps per stage.
- `examples/maps/policy_optimization/eval/stage{1,2,3}/` stores 16 held-out evaluation maps per stage.
- Map files are plain UTF-8 text and may only use `w`, `a`, `g`, `o`, and space.

## Generate The Corpus

```bash
python examples/Policy-Optimization/generate_map_corpus.py --force
```

The generator is deterministic for a fixed seed and only emits maps with exactly one start, one goal, border walls, and a BFS-confirmed safe path.

## Train

```bash
python examples/Policy-Optimization/train_ppo.py
python examples/Policy-Optimization/train_ppo.py --stage 2 --total-steps 300000 --device cpu
python examples/Policy-Optimization/train_ppo.py --resume logs/policy_optimization/checkpoints/latest.pt
```

Artifacts are written to `logs/policy_optimization/`, including `config.json`, `metrics.csv`, and the `best.pt` / `latest.pt` checkpoints.

## One-Command Flow

```bash
python examples/Policy-Optimization/main.py
python examples/Policy-Optimization/main.py --total-steps 300000 --eval-episodes 100
python examples/Policy-Optimization/main.py --resume logs/policy_optimization/checkpoints/best.pt --total-steps 0
```

`main.py` trains the policy, evaluates the resulting checkpoint, and then runs one test episode on a map. Use `--skip-render` for headless runs.
If you prefer using the IDE run button, you can run either `main.py` or `ppo_trainer.py` directly with no required command-line arguments.

## Evaluate

```bash
python examples/Policy-Optimization/evaluate_policy.py --checkpoint logs/policy_optimization/checkpoints/best.pt --stage 3
python examples/Policy-Optimization/evaluate_policy.py --checkpoint logs/policy_optimization/checkpoints/best.pt --stage final --episodes 200
```

## Render

```bash
python examples/Policy-Optimization/render_episode.py --checkpoint logs/policy_optimization/checkpoints/best.pt --stage 3
python examples/Policy-Optimization/render_episode.py --checkpoint logs/policy_optimization/checkpoints/best.pt --map-file examples/maps/policy_optimization/eval/stage3/map_001.txt
```
