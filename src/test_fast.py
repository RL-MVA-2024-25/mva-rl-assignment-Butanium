import numpy as np
from .env_hiv import HIVPatient
from .env_hiv_fast import FastHIVPatient

def compare_envs(env1, env2, rtol=1e-3):
    """Helper function to compare two environments"""
    # Compare observation spaces
    assert env1.observation_space.shape == env2.observation_space.shape
    assert env1.action_space.n == env2.action_space.n
    
    # Compare action sets
    assert np.allclose(env1.action_set, env2.action_set, rtol=rtol)
    
    # Compare bounds
    assert np.allclose(env1.upper, env2.upper, rtol=rtol)
    assert np.allclose(env1.lower, env2.lower, rtol=rtol)

def test_init_parameters():
    """Test different initialization parameters"""
    parameter_combinations = [
        dict(clipping=True, logscale=False, domain_randomization=False),
        dict(clipping=False, logscale=False, domain_randomization=False),
        dict(clipping=True, logscale=True, domain_randomization=False),
        dict(clipping=True, logscale=False, domain_randomization=True),
    ]
    
    for params in parameter_combinations:
        print(f"\nTesting initialization with parameters: {params}")
        env1 = HIVPatient(**params)
        env2 = FastHIVPatient(**params)
        compare_envs(env1, env2)
        print("✓ Basic environment properties match")

def test_reset_modes():
    """Test different reset modes"""
    modes = ["unhealthy", "healthy", "uninfected"]
    
    for mode in modes:
        print(f"\nTesting reset with mode: {mode}")
        # Test with different parameter combinations
        for domain_rand in [False, True]:
            for logscale in [False, True]:
                env1 = HIVPatient(domain_randomization=domain_rand, logscale=logscale)
                env2 = FastHIVPatient(domain_randomization=domain_rand, logscale=logscale)
                
                # Set same random seed for domain randomization
                np.random.seed(42)
                obs1, _ = env1.reset(mode=mode)
                np.random.seed(42)
                obs2, _ = env2.reset(mode=mode)
                
                assert np.allclose(obs1, obs2, rtol=1e-3), \
                    f"Observations don't match for mode={mode}, domain_rand={domain_rand}, logscale={logscale}"
        print(f"✓ Reset mode {mode} works correctly")

def test_step_outputs():
    """Test step function outputs"""
    np.random.seed(42)
    
    # Test with different parameter combinations
    parameter_combinations = [
        dict(clipping=True, logscale=False, domain_randomization=False),
        dict(clipping=False, logscale=True, domain_randomization=True),
    ]
    
    for params in parameter_combinations:
        print(f"\nTesting step outputs with parameters: {params}")
        env1 = HIVPatient(**params)
        env2 = FastHIVPatient(**params)
        
        # Reset environments
        np.random.seed(42)
        obs1, _ = env1.reset()
        np.random.seed(42)
        obs2, _ = env2.reset()
        
        # Test multiple steps
        for action in range(4):  # Test all possible actions
            next_obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
            next_obs2, reward2, terminated2, truncated2, info2 = env2.step(action)
            
            # Compare outputs
            assert np.allclose(next_obs1, next_obs2, rtol=1e-3), \
                f"Observations don't match for action {action}"
            assert np.abs(reward1 - reward2) < 1e-10, \
                f"Rewards don't match for action {action}"
            assert terminated1 == terminated2, \
                f"Terminated flags don't match for action {action}"
            assert truncated1 == truncated2, \
                f"Truncated flags don't match for action {action}"
        print("✓ Step outputs match for all actions")

def run_all_tests():
    """Run all tests"""
    print("Starting test suite...")
    
    print("\n1. Testing initialization parameters")
    test_init_parameters()
    
    print("\n2. Testing reset modes")
    test_reset_modes()
    
    print("\n3. Testing step outputs")
    test_step_outputs()
    
    print("\nAll tests completed successfully! ✓")

if __name__ == "__main__":
    run_all_tests()