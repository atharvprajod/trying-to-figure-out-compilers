"""
Training utilities for RL-based operator fusion optimization.

This module provides functions for training RL agents on the fusion environment.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import pickle
from tqdm import tqdm

from src.rl_fusion.environment.fusion_env import FusionEnv
from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.models.cost_model import SimpleCostModel


class PPOTrainer:
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.
    
    This class implements the PPO training loop for operator fusion.
    """
    
    def __init__(
        self,
        env: FusionEnv,
        policy: PPOPolicy,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            env: The fusion environment
            policy: The policy to train
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            update_epochs: Number of epochs to update the policy
            batch_size: Batch size for training
            device: Device to train on
        """
        self.env = env
        self.policy = policy
        self.device = device
        
        self.policy.to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Memory for trajectory storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []
    
    def collect_rollout(self, num_steps: int = 2048):
        """
        Collect a rollout of trajectories.
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            episode_info: Dictionary with episode information
        """
        # Reset memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []
        
        state = self.env.reset()
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        # Collect trajectories
        for _ in range(num_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action.cpu().numpy()[0])
            
            # Store transition
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
            self.log_probs.append(log_prob.cpu().numpy()[0])
            self.rewards.append(reward)
            self.values.append(value.cpu().numpy()[0])
            self.terminals.append(done)
            
            state = next_state
            
            # Update episode stats
            current_episode_reward += reward
            current_episode_length += 1
            
            if done:
                # End of episode
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                state = self.env.reset()
        
        # Convert to numpy arrays
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.log_probs = np.array(self.log_probs)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)
        self.terminals = np.array(self.terminals)
        
        # Compute returns and advantages
        returns, advantages = self._compute_returns_and_advantages()
        
        # Create training data
        self.train_data = {
            "states": self.states,
            "actions": self.actions,
            "log_probs": self.log_probs,
            "returns": returns,
            "advantages": advantages,
        }
        
        # Episode information
        episode_info = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
        }
        
        return episode_info
    
    def _compute_returns_and_advantages(self):
        """
        Compute returns and advantages for the collected trajectories.
        
        Returns:
            returns: Discounted returns
            advantages: Advantages (returns - values)
        """
        # Compute returns using GAE (Generalized Advantage Estimation)
        returns = np.zeros_like(self.rewards)
        advantages = np.zeros_like(self.rewards)
        
        last_return = 0
        last_advantage = 0
        
        for t in reversed(range(len(self.rewards))):
            # If terminal state, bootstrap is zero
            if self.terminals[t]:
                next_value = 0
                next_advantage = 0
            else:
                next_value = last_return
                next_advantage = last_advantage
            
            # Compute return and advantage
            returns[t] = self.rewards[t] + self.gamma * next_value * (1 - self.terminals[t])
            advantages[t] = returns[t] - self.values[t]
            
            last_return = returns[t]
            last_advantage = advantages[t]
        
        return returns, advantages
    
    def update_policy(self):
        """
        Update the policy using PPO.
        
        Returns:
            loss_info: Dictionary with loss information
        """
        # Extract training data
        states = self.train_data["states"]
        actions = self.train_data["actions"]
        old_log_probs = self.train_data["log_probs"]
        returns = self.train_data["returns"]
        advantages = self.train_data["advantages"]
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Update in batches
            for start_idx in range(0, len(states), self.batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx+self.batch_size]
                
                # Extract batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Compute entropy
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Record losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
        
        # Compute average losses
        num_updates = len(states) // self.batch_size * self.update_epochs
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        loss_info = {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
        }
        
        return loss_info
    
    def train(self, num_iterations: int = 100, steps_per_iteration: int = 2048, save_path: Optional[str] = None):
        """
        Train the policy.
        
        Args:
            num_iterations: Number of training iterations
            steps_per_iteration: Number of steps to collect per iteration
            save_path: Path to save the trained policy
            
        Returns:
            training_info: Dictionary with training information
        """
        # Training statistics
        mean_rewards = []
        mean_lengths = []
        policy_losses = []
        value_losses = []
        entropies = []
        
        best_mean_reward = -float("inf")
        
        # Training loop
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Collect rollout
            episode_info = self.collect_rollout(num_steps=steps_per_iteration)
            
            # Update policy
            loss_info = self.update_policy()
            
            # Record statistics
            mean_rewards.append(episode_info["mean_reward"])
            mean_lengths.append(episode_info["mean_length"])
            policy_losses.append(loss_info["policy_loss"])
            value_losses.append(loss_info["value_loss"])
            entropies.append(loss_info["entropy"])
            
            # Save best model
            if save_path and episode_info["mean_reward"] > best_mean_reward:
                best_mean_reward = episode_info["mean_reward"]
                self.save_policy(os.path.join(save_path, "best_policy.pth"))
            
            # Print progress
            time_elapsed = time.time() - start_time
            print(
                f"Iteration {iteration+1}/{num_iterations} | "
                f"Mean Reward: {episode_info['mean_reward']:.2f} | "
                f"Mean Length: {episode_info['mean_length']:.2f} | "
                f"Policy Loss: {loss_info['policy_loss']:.4f} | "
                f"Value Loss: {loss_info['value_loss']:.4f} | "
                f"Entropy: {loss_info['entropy']:.4f} | "
                f"Time: {time_elapsed:.2f}s"
            )
            
            # Save final model
            if save_path and iteration == num_iterations - 1:
                self.save_policy(os.path.join(save_path, "final_policy.pth"))
        
        # Training information
        training_info = {
            "mean_rewards": mean_rewards,
            "mean_lengths": mean_lengths,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "entropies": entropies,
        }
        
        return training_info
    
    def save_policy(self, path: str):
        """
        Save the policy to a file.
        
        Args:
            path: Path to save the policy
        """
        torch.save(self.policy.state_dict(), path)
    
    def load_policy(self, path: str):
        """
        Load the policy from a file.
        
        Args:
            path: Path to load the policy from
        """
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


class DQNTrainer:
    """
    Trainer for Deep Q-Network (DQN) algorithm.
    
    This class implements the DQN training loop for operator fusion.
    """
    
    def __init__(
        self,
        env: FusionEnv,
        policy: DQNPolicy,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 200000,
        target_update: int = 1000,
        memory_size: int = 10000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the DQN trainer.
        
        Args:
            env: The fusion environment
            policy: The policy to train
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            target_update: Frequency of target network updates
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            device: Device to train on
        """
        self.env = env
        self.policy = policy
        self.device = device
        
        self.policy.to(device)
        
        # Create target network
        self.target_policy = DQNPolicy(
            state_dim=policy.state_dim,
            action_dim=policy.action_dim,
            hidden_dim=policy.hidden_dim,
            use_graph_embedding=policy.use_graph_embedding,
        )
        self.target_policy.to(device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Initialize replay memory
        self.replay_memory = []
        self.memory_position = 0
        
        # Training statistics
        self.steps_done = 0
    
    def get_epsilon(self):
        """
        Get the current exploration rate.
        
        Returns:
            epsilon: Current exploration rate
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)
        return epsilon
    
    def add_to_memory(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.replay_memory) < self.memory_size:
            self.replay_memory.append(None)
        
        self.replay_memory[self.memory_position] = (state, action, reward, next_state, done)
        self.memory_position = (self.memory_position + 1) % self.memory_size
    
    def sample_from_memory(self, batch_size):
        """
        Sample a batch of transitions from the replay memory.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            batch: Batch of transitions
        """
        return np.random.choice(self.replay_memory, batch_size, replace=False)
    
    def optimize_model(self):
        """
        Update the policy using DQN.
        
        Returns:
            loss: The loss value
        """
        if len(self.replay_memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = self.sample_from_memory(self.batch_size)
        
        # Extract batch
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # Compute Q-values
        q_values = self.policy(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_policy(next_state_batch)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(q_values, target_q_values)
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int = 1000, max_steps_per_episode: int = 100, save_path: Optional[str] = None):
        """
        Train the policy.
        
        Args:
            num_episodes: Number of episodes
            max_steps_per_episode: Maximum steps per episode
            save_path: Path to save the trained policy
            
        Returns:
            training_info: Dictionary with training information
        """
        # Training statistics
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        best_mean_reward = -float("inf")
        
        # Training loop
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            
            for step in range(max_steps_per_episode):
                # Get epsilon
                epsilon = self.get_epsilon()
                
                # Select action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy.get_action(state_tensor, epsilon=epsilon).item()
                
                # Take action in environment
                next_state, reward, done, info = self.env.step(action)
                
                # Add to memory
                self.add_to_memory(state, action, reward, next_state, done)
                
                # Optimize model
                loss = self.optimize_model()
                
                # Update statistics
                episode_reward += reward
                episode_loss += loss
                self.steps_done += 1
                
                # Update target network
                if self.steps_done % self.target_update == 0:
                    self.target_policy.load_state_dict(self.policy.state_dict())
                
                # Check if episode is done
                if done:
                    break
                
                state = next_state
            
            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            losses.append(episode_loss / (step + 1))
            
            # Calculate mean reward
            if len(episode_rewards) > 10:
                mean_reward = np.mean(episode_rewards[-10:])
            else:
                mean_reward = np.mean(episode_rewards)
            
            # Save best model
            if save_path and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.save_policy(os.path.join(save_path, "best_policy.pth"))
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Length: {step+1} | "
                    f"Loss: {episode_loss/(step+1):.4f} | "
                    f"Epsilon: {epsilon:.4f}"
                )
            
            # Save final model
            if save_path and episode == num_episodes - 1:
                self.save_policy(os.path.join(save_path, "final_policy.pth"))
        
        # Training information
        training_info = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "losses": losses,
        }
        
        return training_info
    
    def save_policy(self, path: str):
        """
        Save the policy to a file.
        
        Args:
            path: Path to save the policy
        """
        torch.save(self.policy.state_dict(), path)
    
    def load_policy(self, path: str):
        """
        Load the policy from a file.
        
        Args:
            path: Path to load the policy from
        """
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target_policy.load_state_dict(self.policy.state_dict()) 