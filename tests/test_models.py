"""
Tests for the model implementations.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.models.graph_embedding import GraphEmbedding

def test_ppo_policy():
    """Test PPO policy."""
    state_dim = 32
    action_dim = 10
    batch_size = 4
    
    # Create policy
    policy = PPOPolicy(state_dim=state_dim, action_dim=action_dim, use_graph_embedding=False)
    
    # Test forward pass
    state = torch.randn(batch_size, state_dim)
    action_probs, value = policy(state)
    
    # Check shapes
    assert action_probs.shape == (batch_size, action_dim)
    assert value.shape == (batch_size, 1)
    
    # Check probability distribution
    assert torch.all(action_probs >= 0)
    assert torch.all(action_probs <= 1)
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size))
    
    # Test action selection
    action, log_prob, value = policy.get_action(state)
    
    # Check shapes
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert value.shape == (batch_size, 1)

def test_dqn_policy():
    """Test DQN policy."""
    state_dim = 32
    action_dim = 10
    batch_size = 4
    
    # Create policy
    policy = DQNPolicy(state_dim=state_dim, action_dim=action_dim, use_graph_embedding=False)
    
    # Test forward pass
    state = torch.randn(batch_size, state_dim)
    q_values = policy(state)
    
    # Check shapes
    assert q_values.shape == (batch_size, action_dim)
    
    # Test action selection
    action = policy.get_action(state, epsilon=0.0)
    
    # Check shapes
    assert action.shape == (batch_size,)

def test_graph_embedding():
    """Test graph embedding."""
    # Create a simple mock PyTorch Geometric data object
    class MockData:
        def __init__(self):
            self.x = torch.randn(5, 16)  # 5 nodes, 16 features
            self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # 4 edges
            self.batch = torch.zeros(5, dtype=torch.long)  # All nodes in same batch
    
    # Create data object
    data = MockData()
    
    # Create graph embedding
    embedding = GraphEmbedding(
        node_feature_dim=16,
        edge_feature_dim=8,
        hidden_dim=64,
        output_dim=32,
    )
    
    # Test forward pass
    output = embedding(data)
    
    # Check shapes
    assert output.shape == (1, 32)  # 1 graph, 32 features

if __name__ == "__main__":
    test_ppo_policy()
    test_dqn_policy()
    test_graph_embedding()
    print("All tests passed!") 