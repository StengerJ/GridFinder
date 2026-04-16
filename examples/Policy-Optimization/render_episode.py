"""CLI entry point for rendering one greedy PPO episode with pygame."""

import argparse
from pathlib import Path
import path_setup

from config import DEFAULT_MAPS_ROOT
from ppo_trainer import load_checkpoint_model, resolve_device, run_policy_episode


def main():
    """Parses render arguments and replays one checkpoint-driven episode."""

    parser = argparse.ArgumentParser(description="Render one greedy PPO episode.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--map-file", type=Path, default=None)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, device)

    result = run_policy_episode(
        model,
        maps_root=args.maps_root,
        stage=args.stage,
        split="eval",
        seed=args.seed,
        device=device,
        render=True,
        fps=args.fps,
        map_file=args.map_file,
    )
    print(
        f"map_file={result['map_file']} result={result['result']} "
        f"steps={int(result['steps'])} return={result['return']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
