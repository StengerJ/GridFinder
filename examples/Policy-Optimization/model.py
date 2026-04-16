"""Neural network used by PPO for policy logits and state values."""

import torch
from torch import nn
from torch.nn import functional as F

from config import FRAME_STACK, NUM_OBSERVATION_CODES, VIEW_SIZE


class ActorCritic(nn.Module):
    """CNN actor-critic that consumes stacked integer grid observations."""

    def __init__(
        self,
        frame_stack=FRAME_STACK,
        view_size=VIEW_SIZE,
        num_actions=4,
        num_symbols=NUM_OBSERVATION_CODES,
    ):
        super().__init__()
        self.frame_stack = frame_stack
        self.view_size = view_size
        self.num_actions = num_actions
        self.num_symbols = num_symbols

        input_channels = frame_stack * num_symbols
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, frame_stack, view_size, view_size), dtype=torch.long)
            encoded = self._encode(dummy)
            flat_size = self.conv(encoded).flatten(start_dim=1).shape[1]

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def _encode(self, obs):
        """Converts integer observations into one-hot channels for the CNN."""

        if obs.ndim != 4:
            raise ValueError("Expected observations with shape (batch, frames, height, width).")
        obs = obs.long()
        one_hot = F.one_hot(obs, num_classes=self.num_symbols).float()
        batch_size, frames, height, width, num_symbols = one_hot.shape
        return one_hot.permute(0, 1, 4, 2, 3).reshape(batch_size, frames * num_symbols, height, width)

    def forward(self, obs):
        """Produces policy logits and scalar value estimates for a batch."""

        encoded = self._encode(obs)
        hidden = self.shared(self.conv(encoded))
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def act(self, obs, deterministic=False):
        """Samples or greedily chooses actions and returns log-probs and values."""

        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """Evaluates chosen actions for PPO loss computation."""

        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
