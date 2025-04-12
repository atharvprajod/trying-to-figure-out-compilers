#!/usr/bin/env python3
"""
Simple test script to verify the RL-Fusion package functionality.

This script performs basic tests of the environment and models to ensure
they work as expected without requiring TVM.
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

from src.rl_fusion.environment.fusion_env import FusionEnv
from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.models.cost_model import SimpleCostModel
from src.rl_fusion.utils.training import PPOTrainer, DQNTrainer


def test_environment():
    """Test the fusion environment."""
    print("\n=== Testing Fusion Environment ===")
    
    # Create environment
    env = FusionEnv()
    
    # Reset environment
    state = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take a few random actions
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}, Action: {action}, Reward: {reward}, Done: {done}")
        state = next_state
        
        if done:
            print("Episode finished")
            break
    
    print("Environment test passed!")


def test_ppo_policy():
    """Test the PPO policy."""
    print("\n=== Testing PPO Policy ===")
    
    # Create environment
    env = FusionEnv()
    
    # Reset environment to get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create policy
    policy = PPOPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        use_graph_embedding=False,
    )
    
    # Test policy
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action_probs, value = policy(state_tensor)
    
    print(f"Action probabilities: {action_probs}")
    print(f"Value estimate: {value}")
    
    # Test action selection
    action, log_prob, value = policy.get_action(state_tensor)
    
    print(f"Selected action: {action}")
    print(f"Log probability: {log_prob}")
    print(f"Value estimate: {value}")
    
    print("PPO policy test passed!")


def test_dqn_policy():
    """Test the DQN policy."""
    print("\n=== Testing DQN Policy ===")
    
    # Create environment
    env = FusionEnv()
    
    # Reset environment to get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create policy
    policy = DQNPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        use_graph_embedding=False,
    )
    
    # Test policy
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy(state_tensor)
    
    print(f"Q-values: {q_values}")
    
    # Test action selection
    action = policy.get_action(state_tensor, epsilon=0.0)
    
    print(f"Selected action: {action}")
    
    print("DQN policy test passed!")


def test_ppo_training(max_iterations=5):
    """Test PPO training."""
    print("\n=== Testing PPO Training ===")
    
    # Create environment
    env = FusionEnv()
    
    # Reset environment to get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    
    # Create policy
    policy = PPOPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        use_graph_embedding=False,
    )
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        device="cpu",
    )
    
    # Train policy
    start_time = time.time()
    training_info = trainer.train(
        num_iterations=max_iterations,
        steps_per_iteration=128,  # Small for testing
        save_path=None,
    )
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Final mean reward: {training_info['mean_rewards'][-1]:.2f}")
    
    print("PPO training test passed!")


def test_dqn_training(max_episodes=5):
    """Test DQN training."""
    print("\n=== Testing DQN Training ===")
    
    # Create environment
    env = FusionEnv()
    
    # Reset environment to get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    
    # Create policy
    policy = DQNPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        use_graph_embedding=False,
    )
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        policy=policy,
        device="cpu",
        epsilon_decay=100,  # Fast decay for testing
        memory_size=1000,
        batch_size=16,
    )
    
    # Train policy
    start_time = time.time()
    training_info = trainer.train(
        num_episodes=max_episodes,
        max_steps_per_episode=50,  # Small for testing
        save_path=None,
    )
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Final episode reward: {training_info['episode_rewards'][-1]:.2f}")
    
    print("DQN training test passed!")


def test_cost_model():
    """Test the cost model."""
    print("\n=== Testing Cost Model ===")
    
    # Create a simple cost model
    cost_model = SimpleCostModel()
    
    # Create a dummy graph representation
    graph = {
        "num_nodes": 4,
        "num_edges": 3,
        "compute_ops": 2,
        "memory_ops": 1,
        "fusion_count": 2,
    }
    
    # Get predicted time
    predicted_time = cost_model.predict(graph)
    
    print(f"Graph: {graph}")
    print(f"Predicted time: {predicted_time}")
    
    print("Cost model test passed!")


def main():
    """Main function to run tests."""
    print("=== Running RL-Fusion Tests ===")
    
    # Test environment
    test_environment()
    
    # Test policies
    test_ppo_policy()
    test_dqn_policy()
    
    # Test cost model
    test_cost_model()
    
    # Test training (shorter iterations for quick testing)
    test_ppo_training(max_iterations=2)
    test_dqn_training(max_episodes=2)
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    main() 