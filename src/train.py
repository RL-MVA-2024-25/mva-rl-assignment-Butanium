from gymnasium.wrappers import TimeLimit
from gymnasium import Wrapper
import gymnasium as gym
try:
    from .env_hiv import HIVPatient
except ImportError:
    from env_hiv import HIVPatient
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from pathlib import Path
import numpy as np

SAVE_PATH = Path(__file__).parent.parent / "models"
MODEL_NAME = "ppo_hiv_lstm"
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


def preprocess_observation(
    observation,
    last_obs,
    num_steps=None,
    last_action=None,
    stack_size=10,
    add_step=True,
    add_action=True,
    default_action=0,
):
    if add_action:
        if last_action is None:
            last_action = default_action
        observation = np.concatenate([observation, np.array([last_action])], axis=0)
    if add_step:
        if num_steps is None:
            num_steps = 0
        observation = np.concatenate([observation, np.array([num_steps])], axis=0)
    if last_obs is None:
        last_obs = [observation] * stack_size
    last_obs.pop(0)
    last_obs.append(observation)
    return np.stack(last_obs, axis=0), last_obs


class LatestActionWrapper(Wrapper):
    def __init__(self, env, default_action=0):
        super().__init__(env)
        self.default_action = default_action
        # Update observation space to include action
        low = np.append(env.observation_space.low, -float("inf"))
        high = np.append(env.observation_space.high, float("inf"))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.concatenate([obs, np.array([self.default_action])], axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, np.array([action])], axis=0)
        return obs, reward, terminated, truncated, info


class ProjectAgent:
    def __init__(
        self,
        use_lstm=True,
        model_name=None,
        stack_size=10,
        add_step=True,
        deterministic=False,
        add_action=True,
        default_action=0,
    ):
        self.use_lstm = use_lstm
        self.model_name = model_name
        self.model = None
        self.last_state = None
        self.num_steps = 0
        self.stack_size = stack_size
        self.add_step = add_step
        self.prev_obs = None
        self.prev_act = None
        self.deterministic = deterministic
        self.add_action = add_action
        self.default_action = default_action
        print(
            f"Running agent with: MODEL: {self.model_name}\nLSTM: {self.use_lstm}\nSTACK_SIZE: {self.stack_size}\nADD_STEP: {self.add_step}\nDETERMINISTIC: {self.deterministic}\nADD_ACTION: {self.add_action}\nDEFAULT_ACTION: {self.default_action}"
        )

    def reset(self):
        self.num_steps = 0
        self.last_state = None
        self.prev_obs = None

    def preprocess_observation(self, observation):
        observation, self.prev_obs = preprocess_observation(
            observation,
            self.prev_obs,
            self.num_steps,
            self.prev_act,
            stack_size=self.stack_size,
            add_step=self.add_step,
            add_action=self.add_action,
            default_action=self.default_action,
        )
        return observation

    def select_action(self, observation, use_random=False):
        if use_random:
            return self.random_action()
        if self.use_lstm:
            actions, self.last_state = self.model.predict(
                observation, deterministic=self.deterministic, state=self.last_state
            )
            return actions
        return self.model.predict(observation, deterministic=self.deterministic)[0]

    def act(self, observation, use_random=False):
        self.num_steps += 1
        if self.num_steps > 200:
            self.reset()
        observation = self.preprocess_observation(observation)
        act = self.select_action(observation, use_random)
        self.prev_act = act
        return act

    def random_action(self):
        return env.action_space.sample()

    def save(self, path):
        pass

    def load(self):
        if self.model_name is None:
            self.model = PPO.load(SAVE_PATH / MODEL_NAME)
        else:
            if self.use_lstm:
                self.model = RecurrentPPO.load(SAVE_PATH / self.model_name)
                self.last_state = None
            else:
                self.model = PPO.load(SAVE_PATH / self.model_name)
