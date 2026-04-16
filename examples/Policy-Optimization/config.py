"""Configuration values shared across the file-backed PPO example."""

from pathlib import Path


OBSERVATION_CODES = {
    "unknown": 0,
    "empty": 1,
    "wall": 2,
    "hole": 3,
    "goal": 4,
    "agent": 5,
}
NUM_OBSERVATION_CODES = len(OBSERVATION_CODES)
FRAME_STACK = 4
VIEW_SIZE = 11
ACTION_DELTAS = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

DEFAULT_MAPS_ROOT = Path(__file__).resolve().parents[1] / "maps" / "policy_optimization"
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "policy_optimization"
DEFAULT_EVAL_INTERVAL = 20_000
DEFAULT_EVAL_EPISODES = 100
DEFAULT_FORCE_PROMOTION_STEPS = 400_000


class StageConfig:
    """Describes one curriculum stage and the corpus settings tied to it."""

    def __init__(
        self,
        stage_id,
        size,
        wall_probability,
        hole_probability,
        min_goal_distance,
        max_steps,
        train_count=64,
        eval_count=16,
    ):
        self.stage_id = stage_id
        self.size = size
        self.wall_probability = wall_probability
        self.hole_probability = hole_probability
        self.min_goal_distance = min_goal_distance
        self.max_steps = max_steps
        self.train_count = train_count
        self.eval_count = eval_count

    def to_dict(self):
        return {
            "stage_id": self.stage_id,
            "size": self.size,
            "wall_probability": self.wall_probability,
            "hole_probability": self.hole_probability,
            "min_goal_distance": self.min_goal_distance,
            "max_steps": self.max_steps,
            "train_count": self.train_count,
            "eval_count": self.eval_count,
        }


class PPOConfig:
    """Stores the PPO hyperparameters used by the training loop."""

    def __init__(
        self,
        total_steps=1_500_000,
        num_envs=16,
        rollout_steps=256,
        ppo_epochs=4,
        num_minibatches=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        learning_rate=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        eval_interval=DEFAULT_EVAL_INTERVAL,
        eval_episodes=DEFAULT_EVAL_EPISODES,
        target_success_rate=0.8,
        forced_promotion_steps=DEFAULT_FORCE_PROMOTION_STEPS,
        frame_stack=FRAME_STACK,
        view_size=VIEW_SIZE,
    ):
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.target_success_rate = target_success_rate
        self.forced_promotion_steps = forced_promotion_steps
        self.frame_stack = frame_stack
        self.view_size = view_size

    def to_dict(self):
        return {
            "total_steps": self.total_steps,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "ppo_epochs": self.ppo_epochs,
            "num_minibatches": self.num_minibatches,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
            "learning_rate": self.learning_rate,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "target_success_rate": self.target_success_rate,
            "forced_promotion_steps": self.forced_promotion_steps,
            "frame_stack": self.frame_stack,
            "view_size": self.view_size,
        }


STAGE_CONFIGS = {
    1: StageConfig(1, 11, 0.08, 0.02, 4, 80),
    2: StageConfig(2, 15, 0.12, 0.05, 6, 140),
    3: StageConfig(3, 19, 0.16, 0.08, 8, 220),
}
