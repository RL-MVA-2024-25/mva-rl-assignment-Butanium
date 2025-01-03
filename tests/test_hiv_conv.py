import numpy as np
import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env_hiv import SlowHIVPatient
from src.env_hiv_fast import FastHIVPatient

def test_env_speedup():
    """Measure speedup of fast implementation over slow implementation."""
    import time
    
    # Setup environments
    fast_env = FastHIVPatient()
    slow_env = fast_env.to_slow()
    n_steps = 1000
    
    # Time fast implementation
    start_time = time.time()
    for _ in range(n_steps):
        action = np.random.randint(4)
        fast_env.step(action)
    fast_time = time.time() - start_time
    
    # Time slow implementation
    start_time = time.time()
    for _ in range(n_steps):
        action = np.random.randint(4)
        slow_env.step(action)
    slow_time = time.time() - start_time
    
    speedup = slow_time / fast_time
    print(f"\nSpeedup over {n_steps} steps:")
    print(f"Fast implementation: {fast_time:.3f}s")
    print(f"Slow implementation: {slow_time:.3f}s")
    print(f"Speedup factor: {speedup:.1f}x")
    
    assert speedup > 1, "Fast implementation should be faster than slow implementation"


def test_env_equivalence():
    # Create environments
    fast_env = FastHIVPatient(clipping=True, logscale=False)
    slow_env = fast_env.to_slow()
    
    # Test initial states match
    np.testing.assert_array_almost_equal(
        fast_env._get_obs(),
        slow_env.state()
    )
    
    # Test transitions match for each action
    for action in range(4):
        # Reset both envs to same state
        fast_env.reset(mode="unhealthy")
        slow_env.reset(mode="unhealthy")
        
        # Step both environments
        fast_obs, fast_reward, _, _, _ = fast_env.step(action)
        slow_obs, slow_reward, _, _, _ = slow_env.step(action)
        
        # Compare results
        np.testing.assert_allclose(
            fast_obs, 
            slow_obs,
            rtol=1e-1,  # Allow small relative differences
            err_msg=f"Observations differ for action {action}"
        )
        
        np.testing.assert_almost_equal(
            fast_reward,
            slow_reward,
            decimal=4,
            err_msg=f"Rewards differ for action {action}"
        )

def test_env_equivalence_with_options():
    """Test equivalence with different environment options."""
    options = [
        # dict(clipping=True, logscale=True),
        dict(clipping=False, logscale=False),
        # dict(clipping=False, logscale=True),
        dict()
    ] + [dict(domain_randomization=True)] * 10
    
    for opts in options:
        fast_env = FastHIVPatient(**opts)
        slow_env = fast_env.to_slow()
        return_fast = 0
        return_slow = 0
        # perform 20 random steps
        for _ in range(200):
            action = np.random.randint(4)
            fast_obs, fast_reward, _, _, _ = fast_env.step(action)
            slow_obs, slow_reward, _, _, _ = slow_env.step(action)
            return_fast += fast_reward
            return_slow += slow_reward
            np.testing.assert_allclose(
                fast_obs,
                slow_obs,
                rtol=1e-2,  # Allow small relative differences
                err_msg=f"Observations differ with options {opts}"
            )
        
        # Test states match
        np.testing.assert_allclose(
            fast_env._get_obs(),
            slow_env.state(),
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"States differ with options {opts}"
        )
        
        # Test single transition
        action = 1  # Test with one action is sufficient here
        fast_obs, fast_reward, _, _, _ = fast_env.step(action)
        slow_obs, slow_reward, _, _, _ = slow_env.step(action)
        
        np.testing.assert_allclose(
            fast_obs,
            slow_obs,
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"Observations differ with options {opts}"
        )
        
        np.testing.assert_allclose(
            return_fast,
            return_slow,
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"Rewards differ with options {opts}"
        )
