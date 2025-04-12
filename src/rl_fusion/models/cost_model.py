"""
Cost model for predicting the performance of fused operator graphs.

This module provides a simple ML-based cost model that can predict the
execution time of a fused graph based on its structure and operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pickle
import os

from src.rl_fusion.models.graph_embedding import GraphEmbedding


class FusionCostModel(nn.Module):
    """
    Cost model for predicting the runtime of fused operator graphs.
    
    This model takes a graph embedding as input and predicts the execution time.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        hidden_dim: int = 64,
        use_graph_embedding: bool = True,
        graph_embedding_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the cost model.
        
        Args:
            state_dim: Dimension of the state (if not using graph embedding)
            hidden_dim: Dimension of hidden layers
            use_graph_embedding: Whether to use a GNN for graph embedding
            graph_embedding_kwargs: Arguments for the graph embedding model
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_graph_embedding = use_graph_embedding
        
        # Set up the graph embedding network if used
        if use_graph_embedding:
            graph_embedding_kwargs = graph_embedding_kwargs or {}
            self.graph_embedding = GraphEmbedding(
                output_dim=state_dim,
                **graph_embedding_kwargs
            )
        
        # Regression network
        self.regression_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        """
        Forward pass of the cost model.
        
        Args:
            state: Either a graph embedding or a graph data object
            
        Returns:
            predicted_time: Predicted execution time
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
        
        # Get predicted time
        predicted_time = self.regression_network(state_embedding)
        
        # Ensure positive prediction (times are always positive)
        predicted_time = F.softplus(predicted_time)
        
        return predicted_time
    
    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        lr=0.001,
        weight_decay=1e-5,
        patience=10,
        device="cpu",
    ):
        """
        Train the cost model on a dataset of graph-runtime pairs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for L2 regularization
            patience: Early stopping patience
            device: Device to train on
            
        Returns:
            losses: Dictionary of training and validation losses
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                pred_time = self(batch)
                loss = criterion(pred_time, batch.y.view(-1, 1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred_time = self(batch)
                        loss = criterion(pred_time, batch.y.view(-1, 1))
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model("best_cost_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        # Load best model if validation was used
        if val_loader is not None:
            self.load_model("best_cost_model.pth")
        
        return {"train_losses": train_losses, "val_losses": val_losses}
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        self.load_state_dict(torch.load(path))


class SimpleCostModel:
    """
    A simpler, non-neural cost model for fusion performance prediction.
    
    This model uses a simple heuristic or a trained regression model (like XGBoost)
    to predict the performance of fused graphs.
    """
    
    def __init__(self, model_type="heuristic"):
        """
        Initialize the simple cost model.
        
        Args:
            model_type: Type of model to use ("heuristic" or "xgboost")
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == "xgboost":
            try:
                import xgboost as xgb
                self.xgb_available = True
            except ImportError:
                print("XGBoost not available. Falling back to heuristic model.")
                self.model_type = "heuristic"
                self.xgb_available = False
    
    def extract_features(self, graph):
        """
        Extract features from a graph for prediction.
        
        Args:
            graph: The graph to extract features from
            
        Returns:
            features: A dictionary or array of features
        """
        # In a real implementation, this would extract meaningful features
        # from the graph structure, like:
        # - Number of nodes
        # - Number of edges
        # - Types of operations
        # - Memory footprint
        # - Computational complexity
        # etc.
        
        # For now, just return some dummy features
        features = {
            "num_nodes": 4,
            "num_edges": 3,
            "compute_ops": 2,
            "memory_ops": 1,
            "fusion_count": 2,
        }
        
        return features
    
    def heuristic_predict(self, features):
        """
        Make a prediction using a simple heuristic.
        
        Args:
            features: Features extracted from the graph
            
        Returns:
            predicted_time: Predicted execution time
        """
        # Simple heuristic: more fusion = faster (generally)
        # But with diminishing returns and overhead considerations
        
        # Base time for computation
        base_time = features["num_nodes"] * 0.1
        
        # Memory savings from fusion
        memory_savings = features["fusion_count"] * 0.05
        
        # Overhead from very large fused kernels (simplified)
        if features["fusion_count"] > 3:
            overhead = (features["fusion_count"] - 3) * 0.02
        else:
            overhead = 0
        
        # Predicted time
        predicted_time = base_time - memory_savings + overhead
        
        # Ensure positive
        predicted_time = max(0.01, predicted_time)
        
        return predicted_time
    
    def fit(self, X, y):
        """
        Train the cost model on feature-runtime pairs.
        
        Args:
            X: Features extracted from graphs
            y: Measured runtimes
            
        Returns:
            self: The trained model
        """
        if self.model_type == "heuristic":
            # No fitting required for heuristic model
            return self
        
        elif self.model_type == "xgboost" and self.xgb_available:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
            self.model.fit(X, y)
            return self
    
    def predict(self, graph_or_features):
        """
        Predict the execution time of a graph.
        
        Args:
            graph_or_features: Either a graph or pre-extracted features
            
        Returns:
            predicted_time: Predicted execution time
        """
        # Extract features if needed
        if isinstance(graph_or_features, dict):
            features = graph_or_features
        else:
            features = self.extract_features(graph_or_features)
        
        if self.model_type == "heuristic":
            return self.heuristic_predict(features)
        
        elif self.model_type == "xgboost" and self.model is not None:
            # Convert features to the format expected by XGBoost
            x = np.array([list(features.values())])
            return self.model.predict(x)[0]
        
        # Fallback
        return 0.1
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        if self.model_type == "xgboost" and self.model is not None:
            self.model.save_model(path)
        else:
            # Save the simple model (not much to save)
            with open(path, 'wb') as f:
                pickle.dump({"model_type": self.model_type}, f)
    
    def load_model(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        if self.model_type == "xgboost" and self.xgb_available:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)
        else:
            # Load the simple model (not much to load)
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model_type = data["model_type"] 