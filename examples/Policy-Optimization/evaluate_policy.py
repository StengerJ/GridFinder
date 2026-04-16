"""CLI entry point for greedy evaluation of saved PPO checkpoints."""

import argparse
from pathlib import Path

import path_setup
from config import DEFAULT_MAPS_ROOT
from ppo_trainer import load_checkpoint_model, resolve_device, run_policy_evaluation


def main():
    """Parses evaluation arguments and prints aggregate stage metrics."""

    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--stage", type=str, choices=("1", "2", "3", "final"), default="3")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, device)

    stages = (1, 2, 3) if args.stage == "final" else (int(args.stage),)
    for stage in stages:
        metrics = run_policy_evaluation(
            model,
            maps_root=args.maps_root,
            stage=stage,
            split="eval",
            episodes=args.episodes,
            seed=args.seed + stage,
            device=device,
            render=args.render,
            fps=args.fps,
        )
        print(
            f"stage={stage} episodes={int(metrics['episodes'])} avg_return={metrics['avg_return']} "
            f"avg_steps={metrics['avg_steps']} success_rate={metrics['success_rate']} "
            f"hole_rate={metrics['hole_rate']} timeout_rate={metrics['timeout_rate']}"
        )
    return 0


if __name__ == "__main__":
    main()
