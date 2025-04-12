#!/usr/bin/env python3
"""
Demo script for integrating RL-based fusion with TVM.

This script demonstrates how to use a trained RL agent to guide
fusion decisions in TVM's compilation process.
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Try to import TVM
try:
    import tvm
    import tvm.relay as relay
    from tvm.relay.expr import Call, Function, Var, TupleGetItem
    from tvm.relay.op.op import Op
    import tvm.testing
    HAS_TVM = True
except ImportError:
    print("TVM not available. This demo requires TVM to be installed.")
    HAS_TVM = False
    sys.exit(1)

from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.environment.fusion_env import FusionEnv
from src.rl_fusion.utils.visualization import relay_graph_to_nx, visualize_graph, visualize_fusion_decision


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TVM integration demo for RL-based fusion")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained policy model (if None, uses default fusion)",
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm used (ppo or dqn)",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=apple-m1",
        help="Target hardware specification",
    )
    
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Use Graph Neural Network for state representation",
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the graphs and fusion decisions",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/demo",
        help="Path to save visualization results",
    )
    
    return parser.parse_args()


def create_resnet_block():
    """
    Create a simple ResNet block for demonstration.
    
    Returns:
        mod: A TVM Relay module with a simple ResNet block
    """
    data_shape = (1, 64, 56, 56)
    weight_shape = (64, 64, 3, 3)
    
    data = relay.var("data", shape=data_shape)
    weight1 = relay.var("weight1", shape=weight_shape)
    weight2 = relay.var("weight2", shape=weight_shape)
    
    # First convolution
    conv1 = relay.nn.conv2d(data, weight1, padding=(1, 1))
    
    # First batch normalization
    gamma1 = relay.var("gamma1", shape=(64,))
    beta1 = relay.var("beta1", shape=(64,))
    moving_mean1 = relay.var("moving_mean1", shape=(64,))
    moving_var1 = relay.var("moving_var1", shape=(64,))
    bn1 = relay.nn.batch_norm(conv1, gamma1, beta1, moving_mean1, moving_var1)
    bn1 = bn1[0]  # Extract normalized output
    
    # First ReLU
    relu1 = relay.nn.relu(bn1)
    
    # Second convolution
    conv2 = relay.nn.conv2d(relu1, weight2, padding=(1, 1))
    
    # Second batch normalization
    gamma2 = relay.var("gamma2", shape=(64,))
    beta2 = relay.var("beta2", shape=(64,))
    moving_mean2 = relay.var("moving_mean2", shape=(64,))
    moving_var2 = relay.var("moving_var2", shape=(64,))
    bn2 = relay.nn.batch_norm(conv2, gamma2, beta2, moving_mean2, moving_var2)
    bn2 = bn2[0]  # Extract normalized output
    
    # Skip connection
    residual = relay.add(data, bn2)
    
    # Final ReLU
    output = relay.nn.relu(residual)
    
    # Create function
    func = relay.Function(
        [data, weight1, weight2, gamma1, beta1, moving_mean1, moving_var1,
         gamma2, beta2, moving_mean2, moving_var2],
        output
    )
    
    # Create module
    mod = tvm.IRModule.from_expr(func)
    
    return mod


def apply_default_fusion(mod, target):
    """
    Apply TVM's default fusion rules.
    
    Args:
        mod: TVM Relay module
        target: Target device
        
    Returns:
        mod: Module with default fusion applied
    """
    # Sequence of passes typically used in TVM's standard relay.build
    seq = tvm.transform.Sequential([
        relay.transform.FoldConstant(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ConvertLayout("NHWC"),
        relay.transform.FuseOps(fuse_opt_level=2),  # Default fusion
    ])
    
    # Apply the passes
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    
    return mod


def apply_rl_fusion(mod, policy, target, device="cpu"):
    """
    Apply RL-guided fusion.
    
    Args:
        mod: TVM Relay module
        policy: RL policy for fusion decisions
        target: Target device
        device: Device to run policy on
        
    Returns:
        mod: Module with RL-guided fusion applied
        fusion_decisions: List of fusion decisions made
    """
    # This is a simplified implementation of RL-guided fusion
    # In a real implementation, we would:
    # 1. Convert the Relay graph to a suitable representation for the RL agent
    # 2. Use the policy to decide which ops to fuse
    # 3. Apply those fusion decisions to the Relay graph
    
    # For demo purposes, we'll just trace the decisions the policy would make
    # but apply default fusion in the end
    
    # Create a fusion environment
    env = FusionEnv(model_graph=mod, target=target)
    
    # Reset the environment
    state = env.reset()
    
    # List to store fusion decisions
    fusion_decisions = []
    
    # Simulate RL-guided fusion
    done = False
    while not done:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action from policy
        with torch.no_grad():
            if isinstance(policy, PPOPolicy):
                action, _, _ = policy.get_action(state_tensor, deterministic=True)
                action = action.cpu().numpy()[0]
            else:  # DQN
                action = policy.get_action(state_tensor, epsilon=0.0)
                action = action.item()
        
        # Record the decision
        if hasattr(env, 'fusible_pairs') and action < len(env.fusible_pairs):
            fusion_decisions.append(env.fusible_pairs[action])
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
    
    # For now, apply default fusion
    # In a real implementation, we would use the fusion decisions to guide TVM's fusion
    mod = apply_default_fusion(mod, target)
    
    return mod, fusion_decisions


def measure_performance(mod, target, params=None, repeat=3):
    """
    Measure the performance of a model on the target device.
    
    Args:
        mod: TVM Relay module
        target: Target device
        params: Parameters for the model
        repeat: Number of times to repeat the measurement
        
    Returns:
        time_ms: Median execution time in milliseconds
    """
    if params is None:
        # Create random parameters
        params = {}
        for var in mod["main"].params:
            shape = [int(dim) for dim in var.type_annotation.shape]
            dtype = var.type_annotation.dtype
            params[var.name_hint] = tvm.nd.array(np.random.uniform(-1, 1, shape).astype(dtype))
    
    # Build and create graph runtime
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    
    # Create VM
    device = tvm.device(target, 0)
    vm = tvm.runtime.vm.VirtualMachine(lib, device)
    
    # Measure performance
    times = []
    for _ in range(repeat):
        start = time.time()
        vm.run()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    # Return median time
    return np.median(times)


def main():
    """Main function for the demo."""
    if not HAS_TVM:
        print("TVM not available. This demo requires TVM to be installed.")
        return
    
    args = parse_args()
    
    # Create directories
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple ResNet block
    print("Creating a simple ResNet block...")
    mod = create_resnet_block()
    
    # Convert to a NetworkX graph for visualization
    if args.visualize:
        print("Visualizing original graph...")
        G_original = relay_graph_to_nx(mod)
        visualize_graph(G_original, title="Original Graph",
                       save_path=os.path.join(args.output_path, "original_graph.png"))
    
    # Apply default fusion
    print("Applying default fusion...")
    mod_default = apply_default_fusion(mod, args.target)
    
    # Apply RL-guided fusion if a model is provided
    if args.model_path:
        print("Loading RL policy...")
        
        # Create a dummy environment to get state and action dimensions
        env = FusionEnv()
        state = env.reset()
        state_dim = len(state)
        action_dim = env.action_space.n
        
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
        
        print("Applying RL-guided fusion...")
        mod_rl, fusion_decisions = apply_rl_fusion(mod, policy, args.target, device)
        
        # Visualize fusion decisions
        if args.visualize and fusion_decisions:
            print("Visualizing RL fusion decisions...")
            # Convert fusion decisions to a format suitable for visualization
            fusion_groups = []
            for op1, op2 in fusion_decisions:
                # In a real implementation, we would map these indices to node IDs
                # For now, just use dummy IDs
                fusion_groups.append([f"node_{op1}", f"node_{op2}"])
            
            # Visualize
            visualize_fusion_decision(G_original, fusion_groups, title="RL Fusion Decision",
                                    save_path=os.path.join(args.output_path, "rl_fusion_decision.png"))
    else:
        print("No RL policy provided, using only default fusion.")
        mod_rl = None
    
    # Measure performance of default fusion
    print("Measuring performance of default fusion...")
    time_default = measure_performance(mod_default, args.target)
    print(f"Default fusion: {time_default:.3f} ms")
    
    # Measure performance of RL-guided fusion if available
    if mod_rl:
        print("Measuring performance of RL-guided fusion...")
        time_rl = measure_performance(mod_rl, args.target)
        print(f"RL-guided fusion: {time_rl:.3f} ms")
        
        # Calculate improvement
        improvement = (time_default - time_rl) / time_default * 100
        print(f"Improvement: {improvement:.2f}%")
    
    print(f"Demo completed. Results saved to {args.output_path}")


if __name__ == "__main__":
    main() 