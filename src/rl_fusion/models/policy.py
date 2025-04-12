"""
Reinforcement Learning policies for operator fusion decisions.

This module provides policy implementations for both PPO and DQN approaches
to the operator fusion optimization problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from src.rl_fusion.models.graph_embedding import GraphEmbedding


class PPOPolicy(nn.Module):
    """
    Proximal Policy Optimization (PPO) policy for operator fusion.
    
    This model takes a graph embedding as input and outputs:
    1. A policy (probability distribution over actions)
    2. A value estimate (for the critic)
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        hidden_dim: int = 64,
        use_graph_embedding: bool = True,
        graph_embedding_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the PPO policy.
        
        Args:
            state_dim: Dimension of the state (if not using graph embedding)
            action_dim: Dimension of the action space (usually num_fusible_pairs + 1)
            hidden_dim: Dimension of hidden layers
            use_graph_embedding: Whether to use a GNN for graph embedding
            graph_embedding_kwargs: Arguments for the graph embedding model
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_graph_embedding = use_graph_embedding
        
        # Set up the graph embedding network if used
        if use_graph_embedding:
            graph_embedding_kwargs = graph_embedding_kwargs or {}
            self.graph_embedding = GraphEmbedding(
                output_dim=state_dim,
                **graph_embedding_kwargs
            )
        
        # Policy network (actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Value network (critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        """
        Forward pass of the policy.
        
        Args:
            state: Either a graph embedding or a graph data object
            
        Returns:
            action_probs: Probabilities for each action
            value: Value estimate
        """
        if self.use_graph_embedding:
            if hasattr(state, 'x') and hasattr(state, 'edge_index'):
                # If state is a graph data object, embed it
                state_embedding = self.graph_embedding(state)
            else:
                # Assume state is already an embedding
                state_embedding = state
        else:
            # Use the raw state
            state_embedding = state
        
        # Get action logits
        action_logits = self.actor(state_embedding)
        
        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get value estimate
        value = self.critic(state_embedding)
        
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """
        Get an action based on the policy.
        
        Args:
            state: The current state
            deterministic: Whether to select the most likely action
            
        Returns:
            action: The selected action
            action_log_prob: Log probability of the selected action
            value: Value estimate
        """
        # Get action probabilities and value
        action_probs, value = self.forward(state)
        
        if deterministic:
            # Select the most likely action
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        # Get log probability
        action_log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        return action, action_log_prob, value
    
    def evaluate_actions(self, state, actions):
        """
        Evaluate actions based on the current policy.
        
        Args:
            state: The current state
            actions: The actions to evaluate
            
        Returns:
            action_log_probs: Log probabilities of the actions
            values: Value estimates
            entropy: Entropy of the policy
        """
        # Get action probabilities and value
        action_probs, values = self.forward(state)
        
        # Get log probabilities
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        return action_log_probs, values, entropy


class DQNPolicy(nn.Module):
    """
    Deep Q-Network (DQN) policy for operator fusion.
    
    This model takes a graph embedding as input and outputs Q-values
    for each possible action.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 10,
        hidden_dim: int = 64,
        use_graph_embedding: bool = True,
        graph_embedding_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the DQN policy.
        
        Args:
            state_dim: Dimension of the state (if not using graph embedding)
            action_dim: Dimension of the action space (usually num_fusible_pairs + 1)
            hidden_dim: Dimension of hidden layers
            use_graph_embedding: Whether to use a GNN for graph embedding
            graph_embedding_kwargs: Arguments for the graph embedding model
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_graph_embedding = use_graph_embedding
        
        # Set up the graph embedding network if used
        if use_graph_embedding:
            graph_embedding_kwargs = graph_embedding_kwargs or {}
            self.graph_embedding = GraphEmbedding(
                output_dim=state_dim,
                **graph_embedding_kwargs
            )
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        """
        Forward pass of the Q-network.
        
        Args:
            state: Either a graph embedding or a graph data object
            
        Returns:
            q_values: Q-values for each action
        """
        if self.use_graph_embedding:
            if hasattr(state, 'x') and hasattr(state, 'edge_index'):
                # If state is a graph data object, embed it
                state_embedding = self.graph_embedding(state)
            else:
                # Assume state is already an embedding
                state_embedding = state
        else:
            # Use the raw state
            state_embedding = state
        
        # Get Q-values
        q_values = self.q_network(state_embedding)
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get an action based on the epsilon-greedy policy.
        
        Args:
            state: The current state
            epsilon: Exploration probability
            
        Returns:
            action: The selected action
        """
        if np.random.random() < epsilon:
            # Explore: select a random action
            action = torch.randint(0, self.action_dim, (1,))
        else:
            # Exploit: select the action with highest Q-value
            with torch.no_grad():
                q_values = self.forward(state)
                action = torch.argmax(q_values, dim=-1)
        
        return action 