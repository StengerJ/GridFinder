"""Training, evaluation, logging, and checkpoint helpers for PPO."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

import path_setup
from config import DEFAULT_LOG_DIR, PPOConfig, STAGE_CONFIGS
from map_corpus import list_stage_map_files
from model import ActorCritic
from policy_env import FileBackedGridEnv, SyncVectorEnv


class RolloutBuffer:
    """Stores one PPO rollout and computes GAE returns over it."""

    def __init__(self, rollout_steps, num_envs, obs_shape):
        """
        Initializes the rollout buffer with the specified dimensions and data types for observations, actions, log probabilities, rewards, dones, values, advantages, and returns.
        - rollout_steps: The number of steps in each rollout.
        - num_envs: The number of parallel environments in the vectorized setup.
        - obs_shape: The shape of the observations from the environment, used to initialize the observation buffer with the correct dimensions.
        The buffer is designed to efficiently store the data collected during rollouts
        """
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs = np.zeros((rollout_steps, num_envs, *obs_shape), dtype=np.int8)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    def compute_returns(self, last_values, gamma, gae_lambda):
        """Approximates the advantage function at each step of the rollout using Generalized Advantage Estimation (GAE).
        It iterates backward through the rollout, computing the temporal-difference error (delta) at each step and accumulating the advantages using the GAE formula. The returns are then calculated as the sum
        of the advantages and the value estimates, which will be used as targets for the value function during PPO optimization.
        - last_values: The value estimates for the observations at the end of the rollout, used to bootstrap the advantage calculation for the final step.
        - gamma: The discount factor that determines the present value of future rewards.
        - gae_lambda: The GAE lambda parameter that controls the bias-variance tradeoff in the advantage estimation. A value of 0 corresponds to using only the one-step TD error, while a value close to 1 incorporates more of the future rewards into the advantage estimate."""

        last_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def flatten(self):
        """Flattens the time and environment matrix into one vector."""

        return {
            "obs": self.obs.reshape((-1, *self.obs.shape[2:])),
            "actions": self.actions.reshape(-1),
            "log_probs": self.log_probs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }


def resolve_device(requested):
    """Resolves training device to either CPU or GPU"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def append_metrics_row(path, row):
    """Appends one metrics row to the CSV log, creating the header when needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize_episodes(results):
    """Summarizes a list of evaluation episode outcomes into aggregate metrics."""

    count = max(len(results), 1)
    return {
        "episodes": float(len(results)),
        "avg_return": float(np.mean([item["return"] for item in results])) if results else 0.0,
        "avg_steps": float(np.mean([item["steps"] for item in results])) if results else 0.0,
        "success_rate": float(sum(item["result"] == "goal" for item in results) / count),
        "hole_rate": float(sum(item["result"] == "hole" for item in results) / count),
        "timeout_rate": float(sum(item["result"] == "timeout" for item in results) / count),
    }


def run_policy_episode(
    model,
    maps_root=None,
    stage=None,
    split="eval",
    seed=0,
    device=None,
    render=False,
    fps=10.0,
    map_file=None,
):
    """Runs one policy-driven episode and optionally renders it step by step."""

    if map_file is None and (maps_root is None or stage is None):
        raise ValueError("maps_root and stage are required when map_file is not provided.")

    if map_file is None:
        map_paths = list_stage_map_files(maps_root, split, stage)
        max_steps = STAGE_CONFIGS[stage].max_steps
    else:
        map_paths = [Path(map_file)]
        if stage is None:
            raise ValueError("stage is required when using map_file so max_steps can be determined.")
        max_steps = STAGE_CONFIGS[stage].max_steps

    env = FileBackedGridEnv(map_paths, max_steps=max_steps, seed=seed)
    try:
        obs = env.reset()
        done = False
        episode_return = 0.0
        final_info = {"result": "timeout", "steps": 0, "map_file": str(env.current_map_path)}
        while not done:
            if render:
                env.render()
            obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.long, device=device)
            with torch.no_grad():
                action, _, _ = model.act(obs_tensor, deterministic=True)
            obs, reward, done, info = env.step(int(action.item()))
            episode_return += float(reward)
            final_info = info
            if render and fps > 0:
                time.sleep(1.0 / fps)
        if render:
            env.render()
    finally:
        env.close()

    return {
        "return": episode_return,
        "steps": float(final_info["steps"]),
        "result": final_info["result"],
        "map_file": final_info["map_file"],
        "success": float(final_info["result"] == "goal"),
    }


def run_policy_evaluation(
    model,
    maps_root,
    stage,
    split,
    episodes,
    seed,
    device,
    render=False,
    fps=10.0,
    map_file=None,
):
    """Runs greedy evaluation episodes on one stage or a single chosen map file."""

    results = []
    for episode_index in range(episodes):
        result = run_policy_episode(
            model,
            maps_root=maps_root,
            stage=stage,
            split=split,
            seed=seed + episode_index,
            device=device,
            render=render,
            fps=fps,
            map_file=map_file,
        )
        results.append(
            {
                "return": result["return"],
                "steps": result["steps"],
                "result": result["result"],
            }
        )

    summary = summarize_episodes(results)
    summary["stage"] = float(stage)
    return summary


def load_checkpoint_model(checkpoint_path, device):
    """Loads a checkpoint into an ActorCritic instance for eval or rendering."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


class PPOTrainer:
    """Class for PPO training loop, curriculum progression, and checkpointing."""

    def __init__(self, maps_root, log_dir=DEFAULT_LOG_DIR, config=None, device="auto", seed=0, start_stage=1):
        self.maps_root = Path(maps_root)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or PPOConfig()
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.global_step = 0
        self.current_stage = start_stage
        self.stage_steps = 0
        self.best_success_rate = float("-inf")
        self.metrics_path = self.log_dir / "metrics.csv"
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()
        self.train_envs = self._build_vector_env(self.current_stage)

    def _save_config(self):
        """Writes the current seed, stage settings, and PPO hyperparameters to disk."""

        snapshot = {
            "seed": self.seed,
            "ppo": self.config.to_dict(),
            "stages": {stage_id: stage.to_dict() for stage_id, stage in STAGE_CONFIGS.items()},
        }
        (self.log_dir / "config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def _build_vector_env(self, stage):
        """Builds the current stage's synchronous training environment set."""

        map_paths = list_stage_map_files(self.maps_root, "train", stage)
        max_steps = STAGE_CONFIGS[stage].max_steps
        env_fns = []
        for offset in range(self.config.num_envs):
            env_seed = self.seed + offset
            env_fns.append(
                lambda map_paths=map_paths, max_steps=max_steps, env_seed=env_seed: FileBackedGridEnv(
                    map_paths,
                    max_steps=max_steps,
                    seed=env_seed,
                )
            )
        return SyncVectorEnv(env_fns)

    def _set_stage(self, stage):
        """Promotes training to a new stage and rebuilds the vector environments."""

        if stage == self.current_stage:
            return
        self.train_envs.close()
        self.current_stage = stage
        self.stage_steps = 0
        self.train_envs = self._build_vector_env(stage)

    def _collect_rollout(self, next_obs):
        """Collects one fixed-length rollout across every training environment."""

        buffer = RolloutBuffer(self.config.rollout_steps, self.config.num_envs, self.train_envs.observation_space.shape)
        self.model.eval()
        for step in range(self.config.rollout_steps):
            buffer.obs[step] = next_obs
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.long, device=self.device)
            with torch.no_grad():
                actions, log_probs, values = self.model.act(obs_tensor, deterministic=False)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, _ = self.train_envs.step(actions_np)
            buffer.actions[step] = actions_np
            buffer.log_probs[step] = log_probs.cpu().numpy()
            buffer.values[step] = values.cpu().numpy()
            buffer.rewards[step] = rewards
            buffer.dones[step] = dones.astype(np.float32)
            self.global_step += self.config.num_envs
            self.stage_steps += self.config.num_envs

        with torch.no_grad():
            _, last_values = self.model(torch.as_tensor(next_obs, dtype=torch.long, device=self.device))
        buffer.compute_returns(last_values.cpu().numpy(), self.config.gamma, self.config.gae_lambda)
        return buffer, next_obs

    def _update(self, buffer):
        """Runs multiple PPO optimization epochs over the collected rollout."""

        flat = buffer.flatten()
        batch_size = flat["obs"].shape[0]
        if batch_size % self.config.num_minibatches != 0:
            raise ValueError("rollout_steps * num_envs must be divisible by num_minibatches.")

        advantages = flat["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        minibatch_size = batch_size // self.config.num_minibatches
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

        self.model.train()
        indices = np.arange(batch_size)
        for _ in range(self.config.ppo_epochs):
            self.rng.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                batch_index = indices[start:start + minibatch_size]
                obs = torch.as_tensor(flat["obs"][batch_index], dtype=torch.long, device=self.device)
                actions = torch.as_tensor(flat["actions"][batch_index], dtype=torch.long, device=self.device)
                old_log_probs = torch.as_tensor(flat["log_probs"][batch_index], dtype=torch.float32, device=self.device)
                old_values = torch.as_tensor(flat["values"][batch_index], dtype=torch.float32, device=self.device)
                returns = torch.as_tensor(flat["returns"][batch_index], dtype=torch.float32, device=self.device)
                batch_advantages = torch.as_tensor(advantages[batch_index], dtype=torch.float32, device=self.device)

                new_log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_unclipped = (values - returns) ** 2
                value_clipped = old_values + torch.clamp(values - old_values, -self.config.clip_coef, self.config.clip_coef)
                value_loss = 0.5 * torch.max(value_unclipped, (value_clipped - returns) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy_mean.item()))
                stats["approx_kl"].append(float((ratio - 1.0 - log_ratio).mean().item()))

        return {name: float(np.mean(values)) for name, values in stats.items()}

    def _save_checkpoint(self, name):
        """Serializes the model, optimizer, and curriculum state to disk."""

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "stage_steps": self.stage_steps,
            "best_success_rate": self.best_success_rate,
            "seed": self.seed,
            "config": self.config.to_dict(),
        }
        torch.save(payload, self.checkpoint_dir / name)

    def load_checkpoint(self, checkpoint_path):
        """Restores a training run from a saved checkpoint file."""

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_step = int(checkpoint["global_step"])
        self.current_stage = int(checkpoint["current_stage"])
        self.stage_steps = int(checkpoint["stage_steps"])
        self.best_success_rate = float(checkpoint["best_success_rate"])
        self.train_envs.close()
        self.train_envs = self._build_vector_env(self.current_stage)

    def _log_metrics(self, split, stage, metrics):
        """Writes one training or evaluation metrics record to the CSV log."""

        row = {"global_step": self.global_step, "stage": stage, "split": split}
        row.update(metrics)
        append_metrics_row(self.metrics_path, row)

    def learn(self):
        """Executes PPO training until the configured environment-step budget is reached."""

        next_obs = self.train_envs.reset()
        while self.global_step < self.config.total_steps:
            buffer, next_obs = self._collect_rollout(next_obs)
            train_metrics = self._update(buffer)
            self._log_metrics("train_update", self.current_stage, train_metrics)

            if self.global_step % self.config.eval_interval == 0 or self.global_step >= self.config.total_steps:
                eval_metrics = run_policy_evaluation(
                    self.model,
                    maps_root=self.maps_root,
                    stage=self.current_stage,
                    split="eval",
                    episodes=self.config.eval_episodes,
                    seed=self.seed + self.global_step,
                    device=self.device,
                )
                self._log_metrics("eval", self.current_stage, eval_metrics)
                self._save_checkpoint("latest.pt")
                if eval_metrics["success_rate"] > self.best_success_rate:
                    self.best_success_rate = eval_metrics["success_rate"]
                    self._save_checkpoint("best.pt")

                should_force = self.current_stage < 3 and self.stage_steps >= self.config.forced_promotion_steps
                should_promote = self.current_stage < 3 and eval_metrics["success_rate"] >= self.config.target_success_rate
                if should_force or should_promote:
                    self._set_stage(self.current_stage + 1)
                    next_obs = self.train_envs.reset()

        self._save_checkpoint("latest.pt")
        self.train_envs.close()


if __name__ == "__main__":
    """Allows the IDE run button on this file to launch the full PPO pipeline."""

    from main import main as pipeline_main
    pipeline_main()
