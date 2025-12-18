"""
Deep Q-Network (DQN) agent for Kalshi BTC trading.

Implements DQN with experience replay, target network, and epsilon-greedy exploration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import random

from src.common.logger import get_logger


logger = get_logger(__name__)


class QNetwork(nn.Module):
    """
    Simple MLP Q-network.

    Input: observation vector
    Output: Q-values for each action (3 actions: do nothing, buy YES, sell YES)
    """

    def __init__(self, obs_dim: int, hidden_sizes: List[int] = [128, 128]):
        """
        Initialize Q-network.

        Args:
            obs_dim: Dimension of observation vector.
            hidden_sizes: List of hidden layer sizes (default: [128, 128]).
        """
        super(QNetwork, self).__init__()

        layers = []
        input_size = obs_dim

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Output layer (3 actions)
        layers.append(nn.Linear(input_size, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, obs_dim).

        Returns:
            Q-values tensor of shape (batch_size, 3).
        """
        return self.network(x)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions (obs, action, reward, next_obs, done) and supports
    random batch sampling.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            seed: Random seed for sampling.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode terminated.
        """
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones) as numpy arrays.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        obs = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int64)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_obs = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.float32)

        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent with experience replay and target network.
    """

    def __init__(
        self,
        obs_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        replay_buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        hidden_sizes: List[int] = [128, 128],
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize DQN agent.

        Args:
            obs_dim: Dimension of observation vector.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor.
            replay_buffer_size: Size of replay buffer.
            batch_size: Batch size for training.
            target_update_freq: Frequency of target network updates (in steps).
            epsilon_start: Initial epsilon for epsilon-greedy exploration.
            epsilon_end: Final epsilon after decay.
            epsilon_decay_steps: Number of steps to decay epsilon.
            hidden_sizes: Hidden layer sizes for Q-network.
            device: PyTorch device (CPU or CUDA).
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.obs_dim = obs_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks
        self.q_network = QNetwork(obs_dim, hidden_sizes).to(self.device)
        self.target_network = QNetwork(obs_dim, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

        # Training state
        self.step_count = 0
        self.training_losses = []

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            obs: Current observation.
            explore: Whether to use exploration (epsilon-greedy).

        Returns:
            Selected action (0, 1, or 2).
        """
        if explore and random.random() < self.get_epsilon():
            # Random action
            return random.randint(0, 2)

        # Greedy action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            action = q_values.argmax().item()

        return action

    def get_epsilon(self) -> float:
        """
        Get current epsilon value (linear decay).

        Returns:
            Current epsilon value.
        """
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end

        # Linear decay
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.step_count / self.epsilon_decay_steps
        )
        return max(self.epsilon_end, epsilon)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode terminated.
        """
        self.replay_buffer.store(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_obs_tensor = torch.FloatTensor(next_obs).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(obs_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Next Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_tensor)
            next_q_value = next_q_values.max(1)[0]
            target = rewards_tensor + self.gamma * next_q_value * (1 - dones_tensor)

        # Compute loss (Huber loss)
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_value, target)

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update step count
        self.step_count += 1

        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target()

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def update_target(self) -> None:
        """Hard copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug(f"Target network updated at step {self.step_count}")

    def save(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: Path to save file.
        """
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.get_epsilon(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        logger.info(f"Loaded checkpoint from {path}")

