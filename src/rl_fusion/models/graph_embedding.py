"""
Graph Neural Network (GNN) for embedding computation graphs.

This module provides a GNN implementation for embedding TVM computation graphs
into fixed-size vectors that can be used as state representations for RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class GraphEmbedding(nn.Module):
    """
    Graph Neural Network for embedding computation graphs.
    
    This model takes a graph representation of a computation graph (nodes=ops, edges=data flow)
    and embeds it into a fixed-size vector using graph convolutional networks.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 3,
        use_gat: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the GraphEmbedding model.
        
        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of the final graph embedding
            num_layers: Number of graph conv layers
            use_gat: Whether to use Graph Attention Networks instead of GCNs
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        
        # Input projection for node features
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim
            
            if use_gat:
                # Graph Attention Layer
                self.conv_layers.append(
                    GATConv(in_channels, out_channels, heads=4, dropout=dropout, concat=False)
                )
            else:
                # Graph Convolution Layer
                self.conv_layers.append(GCNConv(in_channels, out_channels))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Forward pass of the GNN.
        
        Args:
            data: A PyTorch Geometric data object containing:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]
                - batch: Batch indices for each node [num_nodes]
                
        Returns:
            graph_embedding: The graph embedding [batch_size, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        # Project node features
        x = self.node_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get a graph-level embedding
        x = global_mean_pool(x, data.batch)
        
        # Final projection
        graph_embedding = self.output_proj(x)
        
        return graph_embedding


class RelayGraphToData:
    """
    Utility class to convert TVM Relay graphs to PyTorch Geometric data objects.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 8,
        use_mock: bool = False,
    ):
        """
        Initialize the converter.
        
        Args:
            node_feature_dim: Dimension of node features to create
            edge_feature_dim: Dimension of edge features to create
            use_mock: Whether to use mock data (when TVM is not available)
        """
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.use_mock = use_mock
        
        # Define node type encodings
        self.op_type_to_index = {
            "conv2d": 0,
            "batch_norm": 1,
            "relu": 2,
            "add": 3,
            "multiply": 4,
            "dense": 5,
            "var": 6,
            "constant": 7,
            # Add more op types as needed
        }
        
        # Import necessary modules only if not using mock
        if not use_mock:
            try:
                import tvm
                import tvm.relay as relay
                self.tvm = tvm
                self.relay = relay
            except ImportError:
                print("TVM not available, using mock data.")
                self.use_mock = True
    
    def convert(self, relay_graph):
        """
        Convert a Relay graph to a PyTorch Geometric data object.
        
        Args:
            relay_graph: A TVM Relay graph (Module or Function)
            
        Returns:
            data: A PyTorch Geometric data object
        """
        if self.use_mock:
            return self._create_mock_data()
        
        # Extract nodes and edges from the Relay graph
        nodes, edges = self._extract_graph_structure(relay_graph)
        
        # Create node features
        node_features = self._create_node_features(nodes)
        
        # Create edge indices and features
        edge_indices, edge_features = self._create_edge_data(edges)
        
        # Create PyTorch tensors
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        
        # Create a PyTorch Geometric data object
        from torch_geometric.data import Data
        data = Data(
            x=node_features_tensor,
            edge_index=edge_indices_tensor,
            edge_attr=edge_features_tensor,
        )
        
        return data
    
    def _extract_graph_structure(self, relay_graph):
        """
        Extract nodes and edges from a Relay graph.
        
        This is a placeholder implementation. The actual implementation would
        traverse the Relay graph and extract operators and their connections.
        
        Args:
            relay_graph: A TVM Relay graph
            
        Returns:
            nodes: List of node information
            edges: List of edge information
        """
        # This would require traversing the Relay graph using TVM's APIs
        # For now, return a dummy structure
        nodes = [
            {"id": 0, "op_type": "var", "shape": [1, 3, 224, 224]},
            {"id": 1, "op_type": "conv2d", "shape": [1, 16, 224, 224]},
            {"id": 2, "op_type": "batch_norm", "shape": [1, 16, 224, 224]},
            {"id": 3, "op_type": "relu", "shape": [1, 16, 224, 224]},
        ]
        
        edges = [
            {"src": 0, "dst": 1, "tensor_shape": [1, 3, 224, 224]},
            {"src": 1, "dst": 2, "tensor_shape": [1, 16, 224, 224]},
            {"src": 2, "dst": 3, "tensor_shape": [1, 16, 224, 224]},
        ]
        
        return nodes, edges
    
    def _create_node_features(self, nodes):
        """
        Create feature vectors for each node.
        
        Args:
            nodes: List of node information
            
        Returns:
            node_features: Array of node features
        """
        node_features = np.zeros((len(nodes), self.node_feature_dim))
        
        for i, node in enumerate(nodes):
            # One-hot encode op type
            op_type = node["op_type"]
            if op_type in self.op_type_to_index:
                type_idx = self.op_type_to_index[op_type]
                if type_idx < self.node_feature_dim:
                    node_features[i, type_idx] = 1.0
            
            # Encode tensor shape (e.g., using average dimension size)
            if "shape" in node:
                shape = node["shape"]
                if len(shape) > 0:
                    avg_dim = sum(shape) / len(shape)
                    if self.node_feature_dim > len(self.op_type_to_index):
                        node_features[i, len(self.op_type_to_index)] = avg_dim / 1000.0  # Normalize
        
        return node_features
    
    def _create_edge_data(self, edges):
        """
        Create edge indices and features.
        
        Args:
            edges: List of edge information
            
        Returns:
            edge_indices: Array of edge indices
            edge_features: Array of edge features
        """
        edge_indices = []
        edge_features = np.zeros((len(edges), self.edge_feature_dim))
        
        for i, edge in enumerate(edges):
            # Add edge indices
            edge_indices.append([edge["src"], edge["dst"]])
            
            # Create edge features based on tensor shape
            if "tensor_shape" in edge:
                shape = edge["tensor_shape"]
                if len(shape) > 0:
                    # Feature 1: Size of tensor (number of elements)
                    size = np.prod(shape)
                    edge_features[i, 0] = np.log1p(size) / 20.0  # Log normalized
                    
                    # Feature 2: Number of dimensions
                    edge_features[i, 1] = len(shape) / 10.0
                    
                    # Features 3+: Actual dimensions (up to edge_feature_dim-2)
                    for j, dim in enumerate(shape[:self.edge_feature_dim-2]):
                        edge_features[i, j+2] = dim / 1000.0  # Normalize
        
        # Convert edge indices to the format expected by PyTorch Geometric
        edge_indices = np.array(edge_indices).T if edge_indices else np.zeros((2, 0))
        
        return edge_indices, edge_features
    
    def _create_mock_data(self):
        """
        Create mock data for testing when TVM is not available.
        
        Returns:
            data: A PyTorch Geometric data object with mock data
        """
        # Create random node features
        num_nodes = 4
        node_features = np.random.rand(num_nodes, self.node_feature_dim)
        
        # Create a simple chain graph: 0 -> 1 -> 2 -> 3
        edge_indices = np.array([[0, 1, 2], [1, 2, 3]])
        
        # Create random edge features
        num_edges = edge_indices.shape[1]
        edge_features = np.random.rand(num_edges, self.edge_feature_dim)
        
        # Create PyTorch tensors
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_indices_tensor = torch.tensor(edge_indices, dtype=torch.long)
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float)
        
        # Create a PyTorch Geometric data object
        from torch_geometric.data import Data
        data = Data(
            x=node_features_tensor,
            edge_index=edge_indices_tensor,
            edge_attr=edge_features_tensor,
        )
        
        return data 