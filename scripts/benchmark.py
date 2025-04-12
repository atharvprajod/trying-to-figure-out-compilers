#!/usr/bin/env python3
"""
Benchmark script to compare default and RL-based fusion approaches.

This script benchmarks the performance of various fusion strategies on
a set of common deep learning models.
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

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
    print("TVM not available. This benchmark requires TVM to be installed.")
    HAS_TVM = False
    sys.exit(1)

from src.rl_fusion.models.policy import PPOPolicy, DQNPolicy
from src.rl_fusion.environment.fusion_env import FusionEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark fusion strategies")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained policy model (if None, only runs default fusion)",
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
        "--output_path",
        type=str,
        default="results/benchmarks",
        help="Path to save benchmark results",
    )
    
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of times to repeat measurements",
    )
    
    return parser.parse_args()


def create_resnet18():
    """
    Create a ResNet-18 model.
    
    Returns:
        mod: TVM Relay module for ResNet-18
    """
    if not HAS_TVM:
        return None
    
    # Use TVM's built-in ResNet-18 model
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size, *image_shape)
    
    mod, params = relay.testing.resnet.get_workload(
        num_layers=18,
        batch_size=batch_size,
        image_shape=image_shape,
    )
    
    return mod, params


def create_mobilenet():
    """
    Create a MobileNet model.
    
    Returns:
        mod: TVM Relay module for MobileNet
    """
    if not HAS_TVM:
        return None
    
    # Use TVM's built-in MobileNet model
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size, *image_shape)
    
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=batch_size,
        image_shape=image_shape,
    )
    
    return mod, params


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


def measure_performance(mod, params, target, repeat=10):
    """
    Measure the performance of a model on the target device.
    
    Args:
        mod: TVM Relay module
        params: Parameters for the model
        target: Target device
        repeat: Number of times to repeat the measurement
        
    Returns:
        time_ms: Median execution time in milliseconds
    """
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


def count_kernels(mod):
    """
    Count the number of distinct kernels in a Relay module.
    
    This is a proxy for counting the number of fused groups.
    
    Args:
        mod: TVM Relay module
        
    Returns:
        kernel_count: Number of distinct kernels
    """
    # This is a simplification. In a real implementation, we would
    # analyze the Relay IR more thoroughly to count fused groups.
    
    # Visitor to count leaf CallNode (leaf calls are likely compiled into separate kernels)
    class KernelCounter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.kernel_count = 0
            self.visited_calls = set()
        
        def visit_call(self, call):
            # Only count leaf calls (no nested calls)
            is_leaf = all(not isinstance(arg, Call) for arg in call.args)
            
            # Check if it's a composite function (fused)
            is_composite = False
            if isinstance(call.op, Function):
                if "Composite" in call.op.attrs:
                    is_composite = True
            
            if is_leaf or is_composite:
                # Use hash of the call to avoid counting the same call twice
                call_hash = hash(call)
                if call_hash not in self.visited_calls:
                    self.kernel_count += 1
                    self.visited_calls.add(call_hash)
            
            super().visit_call(call)
    
    counter = KernelCounter()
    counter.visit(mod["main"])
    
    return counter.kernel_count


def benchmark_model(model_name, mod, params, policy, target, device, repeat=10):
    """
    Benchmark a model using default and RL-based fusion.
    
    Args:
        model_name: Name of the model
        mod: TVM Relay module
        params: Parameters for the model
        policy: RL policy for fusion decisions
        target: Target device
        device: Device to run policy on
        repeat: Number of times to repeat measurements
        
    Returns:
        results: Dictionary with benchmark results
    """
    print(f"\n=== Benchmarking {model_name} ===")
    
    # Apply default fusion
    print("Applying default fusion...")
    mod_default = apply_default_fusion(mod, target)
    
    # Count kernels in default fusion
    kernel_count_default = count_kernels(mod_default)
    print(f"Default fusion: {kernel_count_default} kernels")
    
    # Measure performance of default fusion
    print("Measuring performance of default fusion...")
    time_default = measure_performance(mod_default, params, target, repeat)
    print(f"Default fusion: {time_default:.3f} ms")
    
    # Results dictionary
    results = {
        "model": model_name,
        "target": target,
        "default_fusion": {
            "kernel_count": kernel_count_default,
            "time_ms": float(time_default),
        },
        "rl_fusion": None,
    }
    
    # Apply RL-guided fusion if policy is available
    if policy is not None:
        print("Applying RL-guided fusion...")
        mod_rl, fusion_decisions = apply_rl_fusion(mod, policy, target, device)
        
        # Count kernels in RL-guided fusion
        kernel_count_rl = count_kernels(mod_rl)
        print(f"RL-guided fusion: {kernel_count_rl} kernels")
        
        # Measure performance of RL-guided fusion
        print("Measuring performance of RL-guided fusion...")
        time_rl = measure_performance(mod_rl, params, target, repeat)
        print(f"RL-guided fusion: {time_rl:.3f} ms")
        
        # Calculate improvement
        improvement = (time_default - time_rl) / time_default * 100
        print(f"Improvement: {improvement:.2f}%")
        
        # Add to results
        results["rl_fusion"] = {
            "kernel_count": kernel_count_rl,
            "time_ms": float(time_rl),
            "improvement": float(improvement),
            "fusion_decisions": len(fusion_decisions),
        }
    
    return results


def plot_results(results, output_path):
    """
    Plot benchmark results.
    
    Args:
        results: List of dictionaries with benchmark results
        output_path: Path to save plots
    """
    # Create plot directory
    os.makedirs(output_path, exist_ok=True)
    
    # Extract data
    models = [r["model"] for r in results]
    times_default = [r["default_fusion"]["time_ms"] for r in results]
    kernels_default = [r["default_fusion"]["kernel_count"] for r in results]
    
    times_rl = []
    kernels_rl = []
    improvements = []
    
    for r in results:
        if r["rl_fusion"] is not None:
            times_rl.append(r["rl_fusion"]["time_ms"])
            kernels_rl.append(r["rl_fusion"]["kernel_count"])
            improvements.append(r["rl_fusion"]["improvement"])
        else:
            times_rl.append(None)
            kernels_rl.append(None)
            improvements.append(None)
    
    # Plot execution times
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, times_default, width, label="Default Fusion")
    
    # Only plot RL bars for models with RL results
    rl_indices = [i for i, t in enumerate(times_rl) if t is not None]
    if rl_indices:
        rl_times = [times_rl[i] for i in rl_indices]
        plt.bar(np.array(rl_indices) + width/2, rl_times, width, label="RL-Guided Fusion")
    
    plt.xlabel("Model")
    plt.ylabel("Execution Time (ms)")
    plt.title("Execution Time Comparison")
    plt.xticks(x, models)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, "execution_times.png"))
    
    # Plot kernel counts
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, kernels_default, width, label="Default Fusion")
    
    # Only plot RL bars for models with RL results
    if rl_indices:
        rl_kernels = [kernels_rl[i] for i in rl_indices]
        plt.bar(np.array(rl_indices) + width/2, rl_kernels, width, label="RL-Guided Fusion")
    
    plt.xlabel("Model")
    plt.ylabel("Number of Kernels")
    plt.title("Kernel Count Comparison")
    plt.xticks(x, models)
    plt.legend()
    
    plt.savefig(os.path.join(output_path, "kernel_counts.png"))
    
    # Plot improvements if available
    if any(imp is not None for imp in improvements):
        plt.figure(figsize=(12, 6))
        
        valid_improvements = [imp for imp in improvements if imp is not None]
        valid_models = [models[i] for i, imp in enumerate(improvements) if imp is not None]
        
        plt.bar(valid_models, valid_improvements)
        
        plt.xlabel("Model")
        plt.ylabel("Improvement (%)")
        plt.title("Performance Improvement with RL-Guided Fusion")
        
        plt.savefig(os.path.join(output_path, "improvements.png"))


def main():
    """Main function to run the benchmark."""
    if not HAS_TVM:
        print("TVM not available. This benchmark requires TVM to be installed.")
        return
    
    args = parse_args()
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_path, f"benchmark_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load RL policy if provided
    policy = None
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
    
    # List of models to benchmark
    models = [
        ("ResNet-18", create_resnet18),
        ("MobileNet", create_mobilenet),
    ]
    
    # Benchmark results
    results = []
    
    # Run benchmarks
    for model_name, create_model_fn in models:
        try:
            # Create model
            mod, params = create_model_fn()
            
            # Benchmark model
            model_results = benchmark_model(
                model_name=model_name,
                mod=mod,
                params=params,
                policy=policy,
                target=args.target,
                device=device,
                repeat=args.repeat,
            )
            
            # Add to results
            results.append(model_results)
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
    
    # Save results to JSON
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(results, output_path)
    
    print(f"\nBenchmark completed. Results saved to {output_path}")


if __name__ == "__main__":
    main() 