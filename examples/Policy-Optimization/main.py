"""Single entry point that trains, evaluates, and then tests PPO on one map."""

import argparse
from pathlib import Path

import path_setup
from config import DEFAULT_LOG_DIR, DEFAULT_MAPS_ROOT, PPOConfig
from ppo_trainer import (
    PPOTrainer,
    load_checkpoint_model,
    resolve_device,
    run_policy_episode,
    run_policy_evaluation,
)


def resolve_checkpoint_path(log_dir, checkpoint_name):
    """Finds the requested checkpoint and falls back to latest when best is unavailable."""

    checkpoint_dir = Path(log_dir) / "checkpoints"
    requested_path = checkpoint_dir / f"{checkpoint_name}.pt"
    if requested_path.exists():
        return requested_path
    if checkpoint_name == "best":
        fallback_path = checkpoint_dir / "latest.pt"
        if fallback_path.exists():
            return fallback_path
    raise FileNotFoundError(f"Checkpoint not found: {requested_path}")


def print_eval_summary(stage, metrics):
    """Prints one concise evaluation summary line."""

    print(
        f"eval stage={stage} episodes={int(metrics['episodes'])} avg_return={metrics['avg_return']:.3f} "
        f"avg_steps={metrics['avg_steps']:.2f} success_rate={metrics['success_rate']:.3f} "
        f"hole_rate={metrics['hole_rate']:.3f} timeout_rate={metrics['timeout_rate']:.3f}"
    )


def print_test_summary(result, rendered):
    """Prints the outcome of the final single-map test run."""

    prefix = "rendered_test" if rendered else "test"
    print(
        f"{prefix} map_file={result['map_file']} result={result['result']} "
        f"steps={int(result['steps'])} return={result['return']:.3f}"
    )


def main():
    """Trains PPO, evaluates the chosen checkpoint, then runs one test episode."""

    parser = argparse.ArgumentParser(
        description="Train PPO, evaluate the trained policy, and then test it on one map."
    )
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--total-steps", type=int, default=1_500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-stage", type=str, choices=("1", "2", "3", "final"), default="final")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, choices=("best", "latest"), default="best")
    parser.add_argument("--test-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--test-map-file", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    config = PPOConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        eval_episodes=args.eval_episodes,
    )
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

    should_train = args.total_steps > 0
    if should_train:
        print("training policy...")
        trainer.learn()
    elif args.resume is None and not resolve_checkpoint_path(args.log_dir, args.checkpoint).exists():
        raise FileNotFoundError("No checkpoint is available. Train first or pass --resume.")

    checkpoint_path = args.resume if (args.resume is not None and not should_train) else resolve_checkpoint_path(args.log_dir, args.checkpoint)
    device = resolve_device(args.device)
    model, _ = load_checkpoint_model(checkpoint_path, device)

    print(f"using checkpoint={checkpoint_path}")
    eval_stages = (1, 2, 3) if args.eval_stage == "final" else (int(args.eval_stage),)
    for stage in eval_stages:
        metrics = run_policy_evaluation(
            model,
            maps_root=args.maps_root,
            stage=stage,
            split="eval",
            episodes=args.eval_episodes,
            seed=args.seed + stage,
            device=device,
        )
        print_eval_summary(stage, metrics)

    print("testing one map...")
    test_result = run_policy_episode(
        model,
        maps_root=args.maps_root,
        stage=args.test_stage,
        split="eval",
        seed=args.seed + 10_000,
        device=device,
        render=not args.skip_render,
        fps=args.fps,
        map_file=args.test_map_file,
    )
    print_test_summary(test_result, rendered=not args.skip_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
