#!/usr/bin/env python3
"""
Test script for the fusion environment with mock TVM implementation.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.append(os.path.abspath('.'))

# Import the environment
from src.rl_fusion.environment.fusion_env import FusionEnv, HAS_TVM

def test_environment_initialization():
    """Test if the environment can be initialized."""
    env = FusionEnv()
    logger.info(f"Environment initialized. HAS_TVM: {HAS_TVM}")
    return env

def test_environment_reset(env):
    """Test if the environment can be reset."""
    state = env.reset()
    logger.info(f"Environment reset. State shape: {state.shape}")
    logger.info(f"Action space: {env.action_space}")
    return state

def test_environment_step(env, state):
    """Test if the environment can take steps."""
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        logger.info(f"Step {i+1}, Action: {action}, Reward: {reward}, Done: {done}")
        logger.info(f"Info: {info}")
        state = next_state
        
        if done:
            logger.info("Episode finished")
            break
    return state

def test_graph_creation():
    """Test if a graph can be created with mock TVM."""
    env = FusionEnv()
    env._create_test_graph()
    logger.info(f"Test graph created: {type(env.current_graph)}")
    return env.current_graph

def main():
    """Main test function."""
    logger.info("Testing fusion environment with mock TVM implementation")
    
    # Test environment initialization
    env = test_environment_initialization()
    
    # Test environment reset
    state = test_environment_reset(env)
    
    # Test environment step
    test_environment_step(env, state)
    
    # Test graph creation
    graph = test_graph_creation()
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 