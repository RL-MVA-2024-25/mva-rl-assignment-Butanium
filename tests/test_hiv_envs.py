import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.env_hiv import HIVPatient
from src.env_hiv_fast import FastHIVPatient
import pytest

def test_hiv_envs_equivalence():
    # Initialize both environments with the same settings
    env_original = HIVPatient(clipping=True, logscale=False, domain_randomization=False)
    env_fast = FastHIVPatient(clipping=True, logscale=False, domain_randomization=False)
    
    # Test different initial conditions
    for mode in ["unhealthy", "healthy", "uninfected"]:
        # Reset both environments with same mode
        obs1, _ = env_original.reset(mode=mode)
        obs2, _ = env_fast.reset(mode=mode)
        
        # Check initial observations match
        np.testing.assert_allclose(obs1, obs2, rtol=1e-5, 
            err_msg=f"Initial observations don't match for mode {mode}")
        
        # Run both environments through same action sequence
        for action in range(4):  # Test all possible actions
            next_obs1, reward1, done1, _, _ = env_original.step(action)
            next_obs2, reward2, done2, _, _ = env_fast.step(action)
            
            # Check observations match
            np.testing.assert_allclose(next_obs1, next_obs2, rtol=1e-5,
                err_msg=f"Observations don't match for mode {mode}, action {action}")
            
            # Check rewards match
            np.testing.assert_allclose(reward1, reward2, rtol=1e-5,
                err_msg=f"Rewards don't match for mode {mode}, action {action}")
            
            # Check done flags match
            assert done1 == done2, f"Done flags don't match for mode {mode}, action {action}"

def test_hiv_envs_with_options():
    """Test environments with different configuration options"""
    config_combinations = [
        dict(clipping=True, logscale=True),
        dict(clipping=False, logscale=True),
        dict(clipping=True, logscale=False),
        dict(clipping=False, logscale=False),
    ]
    
    for config in config_combinations:
        env_original = HIVPatient(**config)
        env_fast = FastHIVPatient(**config)
        
        obs1, _ = env_original.reset()
        obs2, _ = env_fast.reset()
        
        np.testing.assert_allclose(obs1, obs2, rtol=1e-5,
            err_msg=f"Initial observations don't match with config {config}")
        
        # Test a sequence of random actions
        for _ in range(10):
            action = env_original.action_space.sample()
            next_obs1, reward1, _, _, _ = env_original.step(action)
            next_obs2, reward2, _, _, _ = env_fast.step(action)
            
            np.testing.assert_allclose(next_obs1, next_obs2, rtol=1e-5,
                err_msg=f"Observations don't match with config {config}")
            np.testing.assert_allclose(reward1, reward2, rtol=1e-5,
                err_msg=f"Rewards don't match with config {config}")

def test_domain_randomization():
    """Test that domain randomization produces different but valid results"""
    env_original = HIVPatient(domain_randomization=True)
    env_fast = FastHIVPatient(domain_randomization=True)
    
    # Set same random seed for both
    np.random.seed(42)
    obs1, _ = env_original.reset()
    np.random.seed(42)
    obs2, _ = env_fast.reset()
    
    # With same seed, should get same results
    np.testing.assert_allclose(obs1, obs2, rtol=1e-5,
        err_msg="Domain randomization results don't match with same seed")
    
    # Test that different seeds give different but valid results
    trajectories = []
    for seed in range(5):
        np.random.seed(seed)
        obs, _ = env_fast.reset()
        trajectory = [obs]
        
        for _ in range(10):
            obs, _, _, _, _ = env_fast.step(env_fast.action_space.sample())
            trajectory.append(obs)
        
        trajectories.append(np.array(trajectory))
    
    # Check that different seeds give different trajectories
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            assert not np.allclose(trajectories[i], trajectories[j]), \
                "Different seeds produced identical trajectories"

if __name__ == "__main__":
    test_hiv_envs_equivalence()
    test_hiv_envs_with_options()
    test_domain_randomization()
    print("All tests passed!") 