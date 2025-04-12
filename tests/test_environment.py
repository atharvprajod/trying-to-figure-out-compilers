"""
Tests for the fusion environment.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rl_fusion.environment.fusion_env import FusionEnv

def test_fusion_env_init():
    """Test environment initialization."""
    env = FusionEnv()
    assert env is not None

def test_fusion_env_reset():
    """Test environment reset."""
    env = FusionEnv()
    state = env.reset()
    assert state is not None
    assert isinstance(state, np.ndarray)

def test_fusion_env_step():
    """Test environment step."""
    env = FusionEnv()
    state = env.reset()
    
    # Get action space size
    action_dim = env.action_space.n
    
    # Take a valid action
    action = 0 if action_dim > 0 else None
    if action is not None:
        next_state, reward, done, info = env.step(action)
        
        # Check types
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check shapes
        assert next_state.shape == state.shape

if __name__ == "__main__":
    test_fusion_env_init()
    test_fusion_env_reset()
    test_fusion_env_step()
    print("All tests passed!") 