import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from numba import jit


@jit(nopython=True)
def _der(state, action, params):
    """Compute derivatives for the HIV system."""
    T1, T1star, T2, T2star, V, E = state
    eps1, eps2 = action

    # Unpack parameters
    (
        lambda1,
        d1,
        k1,
        m1,
        lambda2,
        d2,
        k2,
        f,
        m2,
        delta,
        NT,
        c,
        rho1,
        rho2,
        lambdaE,
        bE,
        Kb,
        dE,
        Kd,
        deltaE,
    ) = params

    # Pre-compute common terms
    T1starT2star = T1star + T2star
    k1_eps1 = k1 * (1 - eps1)
    k2_feps1 = k2 * (1 - f * eps1)

    # Compute derivatives
    T1dot = lambda1 - d1 * T1 - k1_eps1 * V * T1
    T1stardot = k1_eps1 * V * T1 - delta * T1star - m1 * E * T1star
    T2dot = lambda2 - d2 * T2 - k2_feps1 * V * T2
    T2stardot = k2_feps1 * V * T2 - delta * T2star - m2 * E * T2star
    Vdot = (
        NT * delta * (1 - eps2) * T1starT2star
        - c * V
        - (rho1 * k1_eps1 * T1 + rho2 * k2_feps1 * T2) * V
    )
    Edot = (
        lambdaE
        + bE * T1starT2star * E / (T1starT2star + Kb)
        - dE * T1starT2star * E / (T1starT2star + Kd)
        - deltaE * E
    )

    return np.array([T1dot, T1stardot, T2dot, T2stardot, Vdot, Edot])


@jit(nopython=True)
def _transition(state, action, params, duration=5.0, step_size=1e-3):
    """Faster transition function using numba."""
    state0 = state.copy()
    nb_steps = int(duration / step_size)

    for _ in range(nb_steps):
        der = _der(state0, action, params)
        state0 += der * step_size

    return state0


class FastHIVPatient(gym.Env):
    """Optimized HIV patient simulator"""

    def __init__(self, clipping=True, logscale=False, domain_randomization=False):
        super().__init__()
        self.spec = EnvSpec("FastHIVPatient-v0")

        # Environment configuration
        self.domain_randomization = domain_randomization
        self.clipping = clipping
        self.logscale = logscale

        # Spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32
        )

        # Pre-compute action set
        self.action_set = np.array([[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]])

        # State bounds
        self.upper = np.array([1e6, 5e4, 3200.0, 80.0, 2.5e5, 353200.0])
        self.lower = np.zeros(6)

        # Reward parameters
        self.Q = 0.1
        self.R1 = 20000.0
        self.R2 = 20000.0
        self.S = 1000.0

        # Initialize state
        self.state_vec = np.zeros(6)
        self._reset_patient_parameters()

    def _reset_patient_parameters(self):
        """Initialize or reset patient parameters."""
        if self.domain_randomization:
            self.k1 = np.random.uniform(5e-7, 8e-7)
            self.k2 = np.random.uniform(0.1e-4, 1.0e-4)
            self.f = np.random.uniform(0.29, 0.34)
        else:
            self.k1 = 8e-7
            self.k2 = 1e-4
            self.f = 0.34

        # Create parameter vector for fast computation
        self.params = np.array(
            [
                1e4,  # lambda1
                1e-2,  # d1
                self.k1,  # k1
                1e-5,  # m1
                31.98,  # lambda2
                1e-2,  # d2
                self.k2,  # k2
                self.f,  # f
                1e-5,  # m2
                0.7,  # delta
                100,  # NT
                13,  # c
                1,  # rho1
                1,  # rho2
                1,  # lambdaE
                0.3,  # bE
                100,  # Kb
                0.25,  # dE
                500,  # Kd
                0.1,  # deltaE
            ]
        )

    def reset(self, *, seed=None, options=None, mode="unhealthy"):
        if mode == "uninfected":
            self.state_vec = np.array([1e6, 0.0, 3198.0, 0.0, 0.0, 10.0])
        elif mode == "healthy":
            self.state_vec = np.array([967839.0, 76.0, 621.0, 6.0, 415.0, 353108.0])
        else:  # unhealthy
            self.state_vec = np.array([163573.0, 11945.0, 5.0, 46.0, 63919.0, 24.0])

        self._reset_patient_parameters()
        return self._get_obs(), {}

    def _get_obs(self):
        """Return the current state with appropriate transformations."""
        state = self.state_vec.copy()
        if self.logscale:
            state = np.log10(state)
        return state

    def step(self, action_idx):
        # Get current state (with clipping if enabled)
        current_state = self.state_vec.copy()

        action = self.action_set[action_idx]

        # Use clipped state for transition
        next_state = _transition(current_state, action, self.params)

        # Compute reward using CURRENT state
        reward = -(
            self.Q * current_state[4]
            + self.R1 * action[0] ** 2
            + self.R2 * action[1] ** 2
            - self.S * current_state[5]
        )

        # Apply clipping to next state
        if self.clipping:
            np.clip(next_state, self.lower, self.upper, out=next_state)

        # Update internal state
        self.state_vec = next_state
        

        return self._get_obs(), reward, False, False, {}
