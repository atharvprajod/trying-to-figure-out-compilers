#!/usr/bin/env python3
"""
Training script for RL-based operator fusion in TVM.

This script trains an RL agent to make fusion decisions for TVM graphs.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rl_fusion.environment.fusion_env import FusionEnv
from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.models.cost_model import SimpleCostModel
from src.rl_fusion.utils.training import PPOTrainer, DQNTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent for operator fusion in TVM")
    
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm to use (ppo or dqn)",
    )
    
    parser.add_argument(
        "--model_graph",
        type=str,
        default=None,
        help="Path to model graph in Relay IR (if None, uses a test graph)",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=apple-m1",
        help="Target hardware specification",
    )
    
    parser.add_argument(
        "--use_cost_model",
        action="store_true",
        help="Use cost model for rewards instead of actual measurements",
    )
    
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Use Graph Neural Network for state representation",
    )
    
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations (for PPO) or episodes (for DQN)",
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        default="results",
        help="Path to save results",
    )
    
    return parser.parse_args()


def main():
    """Main function to train the RL agent."""
    args = parse_args()
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_path, f"{args.algorithm}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = FusionEnv(
        model_graph=args.model_graph,
        target=args.target,
        use_cost_model=args.use_cost_model,
    )
    
    # Reset environment to get action and state dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create policy
    if args.algorithm == "ppo":
        policy = PPOPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_graph_embedding=args.use_gnn,
        )
        
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            policy=policy,
            device=device,
        )
        
        # Train policy
        training_info = trainer.train(
            num_iterations=args.num_iterations,
            steps_per_iteration=512,  # Smaller for testing
            save_path=save_path,
        )
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_info["mean_rewards"])
        plt.title("Mean Reward")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        
        plt.subplot(2, 2, 2)
        plt.plot(training_info["mean_lengths"])
        plt.title("Mean Episode Length")
        plt.xlabel("Iteration")
        plt.ylabel("Length")
        
        plt.subplot(2, 2, 3)
        plt.plot(training_info["policy_losses"])
        plt.title("Policy Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        
        plt.subplot(2, 2, 4)
        plt.plot(training_info["value_losses"])
        plt.title("Value Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "training_curves.png"))
        
    elif args.algorithm == "dqn":
        policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_graph_embedding=args.use_gnn,
        )
        
        # Create trainer
        trainer = DQNTrainer(
            env=env,
            policy=policy,
            device=device,
        )
        
        # Train policy
        training_info = trainer.train(
            num_episodes=args.num_iterations,
            max_steps_per_episode=100,
            save_path=save_path,
        )
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_info["episode_rewards"])
        plt.title("Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(2, 2, 2)
        plt.plot(training_info["episode_lengths"])
        plt.title("Episode Length")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        
        plt.subplot(2, 2, 3)
        plt.plot(training_info["losses"])
        plt.title("Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        plt.subplot(2, 2, 4)
        # Plot moving average of rewards
        window_size = 10
        if len(training_info["episode_rewards"]) >= window_size:
            moving_avg = np.convolve(
                training_info["episode_rewards"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            plt.plot(moving_avg)
            plt.title(f"Moving Average Reward (window={window_size})")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "training_curves.png"))
    
    print(f"Training completed. Results saved to {save_path}")


if __name__ == "__main__":
    main() 