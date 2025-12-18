"""
Smoke tests for DQN training.

Tests that training completes without crashing and produces valid outputs.
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.data_loader import load_candles
from agent.dqn import DQNAgent, ReplayBuffer, QNetwork
from scripts.train_dqn import train_dqn


def test_q_network():
    """Test that Q-network can be created and forward pass works."""
    obs_dim = 64
    network = QNetwork(obs_dim, hidden_sizes=[128, 128])

    # Test forward pass
    obs = np.random.randn(1, obs_dim).astype(np.float32)
    import torch
    obs_tensor = torch.FloatTensor(obs)
    q_values = network(obs_tensor)

    assert q_values.shape == (1, 3)  # 3 actions
    assert not torch.isnan(q_values).any()


def test_replay_buffer():
    """Test that replay buffer stores and samples correctly."""
    buffer = ReplayBuffer(capacity=100, seed=42)

    # Store some transitions
    obs_dim = 64
    for i in range(10):
        obs = np.random.randn(obs_dim).astype(np.float32)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        buffer.store(obs, action=0, reward=1.0, next_obs=next_obs, done=False)

    assert len(buffer) == 10

    # Sample batch
    obs_batch, actions, rewards, next_obs_batch, dones = buffer.sample(5)

    assert obs_batch.shape[0] == 5
    assert actions.shape[0] == 5
    assert rewards.shape[0] == 5


def test_dqn_agent():
    """Test that DQN agent can be created and basic methods work."""
    obs_dim = 64
    agent = DQNAgent(
        obs_dim=obs_dim,
        learning_rate=1e-3,
        gamma=0.99,
        replay_buffer_size=1000,
        batch_size=32,
        seed=42,
    )

    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action = agent.select_action(obs, explore=True)
    assert action in [0, 1, 2]

    # Test epsilon
    epsilon = agent.get_epsilon()
    assert 0 <= epsilon <= 1

    # Test storing transitions
    next_obs = np.random.randn(obs_dim).astype(np.float32)
    agent.store(obs, action, reward=1.0, next_obs=next_obs, done=False)

    # Test training (should return None if buffer too small)
    loss = agent.train_step()
    # Loss might be None if buffer too small, that's OK


def test_dqn_training_smoke():
    """Test that a small training run completes without crashing."""
    # Check if synthetic data exists
    data_path = Path("data/btc_synthetic.csv")
    if not data_path.exists():
        pytest.skip("Synthetic data not found, skipping training smoke test")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "models"
        log_csv = Path(tmpdir) / "train_metrics.csv"

        # Run small training
        train_dqn(
            data_path=str(data_path),
            start_date="2025-12-10",
            end_date="2025-12-10",
            total_steps=50,  # Very small for smoke test
            seed=42,
            save_dir=str(save_dir),
            log_csv=str(log_csv),
            checkpoint_every=1000,  # Won't trigger with 50 steps
        )

        # Check that model was saved
        model_path = save_dir / "dqn_last.pt"
        assert model_path.exists(), "Model checkpoint should be created"

        # Check that metrics CSV was created
        assert log_csv.exists(), "Training metrics CSV should be created"

        # Check metrics content
        metrics_df = pd.read_csv(log_csv)
        assert len(metrics_df) > 0, "Metrics should have at least one row"
        assert "episode" in metrics_df.columns
        assert "return" in metrics_df.columns
        assert "epsilon" in metrics_df.columns

        # Check that values are finite
        assert np.isfinite(metrics_df["return"]).all(), "Returns should be finite"
        assert np.isfinite(metrics_df["epsilon"]).all(), "Epsilon should be finite"


def test_dqn_save_load():
    """Test that DQN agent can save and load checkpoints."""
    obs_dim = 64
    agent1 = DQNAgent(obs_dim=obs_dim, seed=42)

    # Store some transitions and train a bit
    obs = np.random.randn(obs_dim).astype(np.float32)
    for i in range(100):
        action = agent1.select_action(obs, explore=True)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        agent1.store(obs, action, reward=1.0, next_obs=next_obs, done=False)
        obs = next_obs

    # Train a few steps
    for _ in range(10):
        agent1.train_step()

    # Save
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name

    try:
        agent1.save(save_path)

        # Load into new agent
        agent2 = DQNAgent(obs_dim=obs_dim, seed=42)
        agent2.load(save_path)

        # Check that step counts match
        assert agent2.step_count == agent1.step_count

        # Check that networks have same weights
        import torch
        for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
            assert torch.allclose(p1, p2)

    finally:
        Path(save_path).unlink()

