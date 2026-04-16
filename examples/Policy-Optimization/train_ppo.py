"""CLI entry point for training the file-backed PPO example."""

import argparse
from pathlib import Path

import path_setup
from config import DEFAULT_LOG_DIR, DEFAULT_MAPS_ROOT, PPOConfig
from ppo_trainer import PPOTrainer


def main():
    """Parses training arguments and launches PPO learning."""

    parser = argparse.ArgumentParser(description="Train PPO on the file-backed hidden-goal grid corpus.")
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = PPOConfig(total_steps=args.total_steps, num_envs=args.num_envs)
    trainer = PPOTrainer(
        maps_root=args.maps_root,
        log_dir=args.log_dir,
        config=config,
        device=args.device,
        seed=args.seed,
        start_stage=args.stage,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    trainer.learn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
