"""
Visualization utilities for RL-based operator fusion.

This module provides functions for visualizing computational graphs and fusion decisions.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json

# Try to import TVM; if not available, provide a mock implementation for testing
try:
    import tvm
    import tvm.relay as relay
    HAS_TVM = True
except ImportError:
    HAS_TVM = False


def relay_graph_to_nx(relay_expr) -> nx.DiGraph:
    """
    Convert a Relay expression/module to a NetworkX graph.
    
    Args:
        relay_expr: A Relay expression or module
        
    Returns:
        G: A NetworkX directed graph
    """
    if not HAS_TVM:
        # Create a dummy graph for testing
        return create_dummy_graph()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Visit counter for unique node IDs
    visit_counter = [0]
    
    # Dictionary to map Relay expressions to node IDs
    expr_to_id = {}
    
    def visit_node(expr):
        """Recursively visit nodes in the Relay expression."""
        if expr in expr_to_id:
            return expr_to_id[expr]
        
        node_id = f"node_{visit_counter[0]}"
        visit_counter[0] += 1
        
        # Extract node type and attributes
        node_type = expr.__class__.__name__
        attrs = {}
        
        if isinstance(expr, relay.expr.Call):
            # For Call nodes, extract the operator name
            if isinstance(expr.op, relay.op.Op):
                op_name = expr.op.name
            else:
                op_name = "unknown_op"
            
            # Add node to graph
            attrs["op_name"] = op_name
            G.add_node(node_id, type=node_type, attrs=attrs, label=op_name)
            
            # Recursively process arguments
            for arg in expr.args:
                arg_id = visit_node(arg)
                G.add_edge(arg_id, node_id)
            
        elif isinstance(expr, relay.expr.Var):
            # For variables
            name = expr.name_hint if hasattr(expr, "name_hint") else "var"
            attrs["name"] = name
            G.add_node(node_id, type=node_type, attrs=attrs, label=name)
            
        elif isinstance(expr, relay.expr.Constant):
            # For constants
            shape = expr.data.shape
            attrs["shape"] = shape
            G.add_node(node_id, type=node_type, attrs=attrs, label=f"const{shape}")
            
        elif isinstance(expr, relay.expr.TupleGetItem):
            # For tuple get item
            attrs["index"] = expr.index
            G.add_node(node_id, type=node_type, attrs=attrs, label=f"tuple_get[{expr.index}]")
            
            # Recursively process tuple
            tuple_id = visit_node(expr.tuple_value)
            G.add_edge(tuple_id, node_id)
            
        else:
            # For other types of expressions
            G.add_node(node_id, type=node_type, attrs=attrs, label=node_type)
        
        # Store mapping from expression to node ID
        expr_to_id[expr] = node_id
        
        return node_id
    
    # Start visiting from the Relay expression
    if isinstance(relay_expr, relay.ir.module.IRModule):
        # If it's a module, get the main function
        main_fn = relay_expr["main"]
        visit_node(main_fn.body)
    else:
        # If it's an expression
        visit_node(relay_expr)
    
    return G


def create_dummy_graph() -> nx.DiGraph:
    """
    Create a dummy graph for testing when TVM is not available.
    
    Returns:
        G: A NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Create a simple graph: input -> conv -> bn -> relu -> output
    G.add_node("node_0", type="Var", attrs={"name": "data"}, label="data")
    G.add_node("node_1", type="Call", attrs={"op_name": "nn.conv2d"}, label="conv2d")
    G.add_node("node_2", type="Call", attrs={"op_name": "nn.batch_norm"}, label="batch_norm")
    G.add_node("node_3", type="Call", attrs={"op_name": "nn.relu"}, label="relu")
    
    G.add_edge("node_0", "node_1")
    G.add_edge("node_1", "node_2")
    G.add_edge("node_2", "node_3")
    
    return G


def visualize_graph(G: nx.DiGraph, title: str = "Computation Graph", figsize: Tuple[int, int] = (10, 8), 
                    node_size: int = 800, save_path: Optional[str] = None):
    """
    Visualize a computation graph.
    
    Args:
        G: A NetworkX directed graph
        title: Title of the plot
        figsize: Figure size
        node_size: Size of the nodes
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=figsize)
    
    # Create position layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node labels
    labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, node_color="skyblue", 
            node_size=node_size, font_size=10, arrows=True)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()


def visualize_fusion_decision(G: nx.DiGraph, fusion_groups: List[List[str]], 
                              title: str = "Fusion Decision", figsize: Tuple[int, int] = (10, 8),
                              node_size: int = 800, save_path: Optional[str] = None):
    """
    Visualize a fusion decision.
    
    Args:
        G: A NetworkX directed graph
        fusion_groups: List of lists, where each inner list contains node IDs in a fusion group
        title: Title of the plot
        figsize: Figure size
        node_size: Size of the nodes
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=figsize)
    
    # Create position layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node labels
    labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}
    
    # Assign colors to fusion groups
    color_list = plt.cm.tab10.colors
    node_colors = {}
    
    for i, group in enumerate(fusion_groups):
        color_idx = i % len(color_list)
        for node in group:
            node_colors[node] = color_list[color_idx]
    
    # Assign default color to ungrouped nodes
    for node in G.nodes():
        if node not in node_colors:
            node_colors[node] = "lightgray"
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=labels, 
            node_color=[node_colors[node] for node in G.nodes()],
            node_size=node_size, font_size=10, arrows=True)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()


def plot_training_curves(training_info: Dict, title: str = "Training Curves", 
                         figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
    """
    Plot training curves.
    
    Args:
        training_info: Dictionary with training information
        title: Title of the plot
        figsize: Figure size
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=figsize)
    
    # Determine if it's PPO or DQN based on keys
    is_ppo = "mean_rewards" in training_info
    
    if is_ppo:
        # PPO curves
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
    else:
        # DQN curves
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
        window_size = min(10, len(training_info["episode_rewards"]))
        if window_size > 0:
            moving_avg = np.convolve(
                training_info["episode_rewards"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            plt.plot(moving_avg)
            plt.title(f"Moving Average Reward (window={window_size})")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()


def compare_performance(results_files: List[str], metric: str = "mean_reward", 
                        title: str = "Performance Comparison", figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None):
    """
    Compare performance across different evaluation results.
    
    Args:
        results_files: List of paths to result JSON files
        metric: Metric to compare
        title: Title of the plot
        figsize: Figure size
        save_path: Path to save the figure (if None, just displays)
    """
    plt.figure(figsize=figsize)
    
    data = []
    labels = []
    
    for file_path in results_files:
        try:
            with open(file_path, "r") as f:
                results = json.load(f)
            
            # Extract relevant metrics
            rl_value = results["rl_metrics"].get(metric, 0)
            default_value = results.get("default_metrics", {}).get(metric, None)
            
            # Extract algorithm name from file or results
            algorithm = results["args"].get("algorithm", os.path.basename(file_path).split("_")[1])
            
            # Add to data
            data.append((algorithm, rl_value, default_value))
            labels.append(algorithm)
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error loading results from {file_path}: {e}")
    
    # Plot the data
    x = np.arange(len(labels))
    width = 0.35
    
    rl_values = [d[1] for d in data]
    plt.bar(x - width/2, rl_values, width, label="RL Policy")
    
    default_values = [d[2] for d in data if d[2] is not None]
    if default_values:
        # Only add default bars if we have default values
        default_x = [i for i, d in enumerate(data) if d[2] is not None]
        plt.bar(np.array(default_x) + width/2, default_values, width, label="Default Policy")
    
    plt.xlabel("Algorithm")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show() 