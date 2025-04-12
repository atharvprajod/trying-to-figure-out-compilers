#!/usr/bin/env python3
"""
Evaluation script for RL-based operator fusion in TVM.

This script evaluates a trained RL agent's fusion decisions for TVM graphs.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rl_fusion.environment.fusion_env import FusionEnv
from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent for operator fusion in TVM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained policy model",
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm used (ppo or dqn)",
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
        action="store_false",
        help="Use actual measurements instead of cost model for evaluation",
    )
    
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Use Graph Neural Network for state representation",
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/evaluation",
        help="Path to save evaluation results",
    )
    
    parser.add_argument(
        "--compare_default",
        action="store_true",
        help="Compare with TVM's default fusion strategy",
    )
    
    return parser.parse_args()


def evaluate_policy(env, policy, num_episodes=10, deterministic=True, device="cpu"):
    """
    Evaluate a policy on the fusion environment.
    
    Args:
        env: The fusion environment
        policy: The policy to evaluate
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        device: Device to run on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    fusion_counts = []
    runtimes = []
    
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        start_time = time.time()
        done = False
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                if isinstance(policy, PPOPolicy):
                    action, _, _ = policy.get_action(state_tensor, deterministic=deterministic)
                    action = action.cpu().numpy()[0]
                else:  # DQN
                    action = policy.get_action(state_tensor, epsilon=0.0 if deterministic else 0.1)
                    action = action.item()
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
        
        end_time = time.time()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        fusion_counts.append(info.get("fusion_count", 0))
        runtimes.append(end_time - start_time)
    
    # Compute metrics
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_fusion_count": float(np.mean(fusion_counts)),
        "mean_runtime": float(np.mean(runtimes)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "fusion_counts": fusion_counts,
        "runtimes": runtimes,
    }
    
    return metrics


def evaluate_default_fusion(env, num_episodes=10):
    """
    Evaluate TVM's default fusion strategy.
    
    Args:
        env: The fusion environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # This is a placeholder. In a real implementation,
    # we would run TVM's default fusion pass on the same model graph.
    
    # For now, we simulate it by always taking the "fuse everything" action
    episode_rewards = []
    episode_lengths = []
    fusion_counts = []
    runtimes = []
    
    for _ in tqdm(range(num_episodes), desc="Evaluating default"):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        start_time = time.time()
        done = False
        
        while not done:
            # Take "fuse everything" action (always choose first fusion option)
            action = 0
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            steps += 1
        
        end_time = time.time()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        fusion_counts.append(info.get("fusion_count", 0))
        runtimes.append(end_time - start_time)
    
    # Compute metrics
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_fusion_count": float(np.mean(fusion_counts)),
        "mean_runtime": float(np.mean(runtimes)),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "fusion_counts": fusion_counts,
        "runtimes": runtimes,
    }
    
    return metrics


def main():
    """Main function to evaluate the RL agent."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.output_path, exist_ok=True)
    
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
    else:  # args.algorithm == "dqn"
        policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            use_graph_embedding=args.use_gnn,
        )
    
    # Load policy
    policy.load_state_dict(torch.load(args.model_path, map_location=device))
    policy.to(device)
    policy.eval()
    
    print(f"Loaded policy from {args.model_path}")
    
    # Evaluate policy
    print("Evaluating RL policy...")
    rl_metrics = evaluate_policy(
        env=env,
        policy=policy,
        num_episodes=args.num_episodes,
        device=device,
    )
    
    print(f"RL Policy - Mean Reward: {rl_metrics['mean_reward']:.4f} ± {rl_metrics['std_reward']:.4f}")
    print(f"RL Policy - Mean Fusion Count: {rl_metrics['mean_fusion_count']:.2f}")
    print(f"RL Policy - Mean Runtime: {rl_metrics['mean_runtime']:.4f}s")
    
    # Compare with default fusion if requested
    if args.compare_default:
        print("Evaluating default fusion strategy...")
        default_metrics = evaluate_default_fusion(
            env=env,
            num_episodes=args.num_episodes,
        )
        
        print(f"Default - Mean Reward: {default_metrics['mean_reward']:.4f} ± {default_metrics['std_reward']:.4f}")
        print(f"Default - Mean Fusion Count: {default_metrics['mean_fusion_count']:.2f}")
        print(f"Default - Mean Runtime: {default_metrics['mean_runtime']:.4f}s")
        
        # Calculate improvement
        reward_improvement = ((rl_metrics["mean_reward"] - default_metrics["mean_reward"]) / 
                             abs(default_metrics["mean_reward"]) * 100)
        runtime_improvement = ((default_metrics["mean_runtime"] - rl_metrics["mean_runtime"]) / 
                              default_metrics["mean_runtime"] * 100)
        
        print(f"Reward Improvement: {reward_improvement:.2f}%")
        print(f"Runtime Improvement: {runtime_improvement:.2f}%")
    
    # Save results
    results = {
        "rl_metrics": rl_metrics,
        "args": vars(args),
    }
    
    if args.compare_default:
        results["default_metrics"] = default_metrics
        results["improvements"] = {
            "reward_improvement": float(reward_improvement),
            "runtime_improvement": float(runtime_improvement),
        }
    
    output_file = os.path.join(args.output_path, f"eval_{args.algorithm}_{int(time.time())}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main() 