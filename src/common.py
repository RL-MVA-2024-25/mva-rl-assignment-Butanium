from functools import partial
from typing import Any, Dict, List, Optional, Union
import os
import warnings

import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.wrappers import (
    FrameStackObservation,
    TimeAwareObservation,
    TimeLimit,
    TransformObservation,
    TransformReward,
)
import numpy as np
import torch as th
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EventCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    sync_envs_normalization,
)
from wandb.integration.sb3 import WandbCallback

try:
    from src.env_hiv import SlowHIVPatient
    from src.env_hiv_fast import FastHIVPatient
    from src.toolbox import plot_scatters, str_of_eval
except ImportError:
    try:
        from env_hiv import SlowHIVPatient
        from env_hiv_fast import FastHIVPatient
        from toolbox import plot_scatters, str_of_eval
    except ImportError:
        from .env_hiv import SlowHIVPatient
        from .env_hiv_fast import FastHIVPatient
        from .toolbox import plot_scatters, str_of_eval


class LatestActionWrapper(Wrapper):
    def __init__(self, env, default_action=0, one_hot_action=False):
        super().__init__(env)
        self.default_action = default_action
        self.one_hot_action = one_hot_action
        # Update observation space to include action
        if one_hot_action:
            low = np.concatenate(
                [env.observation_space.low, np.zeros(env.action_space.n)],
                dtype=env.observation_space.dtype,
            )
            high = np.concatenate(
                [env.observation_space.high, np.ones(env.action_space.n)],
                dtype=env.observation_space.dtype,
            )
        else:
            low = np.append(
                env.observation_space.low, 0, dtype=env.observation_space.dtype
            )
            high = np.append(
                env.observation_space.high,
                env.action_space.n,
                dtype=env.observation_space.dtype,
            )
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def preprocess_action(self, action):
        if self.one_hot_action:
            actions = np.zeros(self.env.action_space.n)
            actions[action] = 1
            return actions
        return np.array([action])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.concatenate([obs, self.preprocess_action(self.default_action)], axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, self.preprocess_action(action)], axis=0)
        return obs, reward, terminated, truncated, info


class NormalizeRewardbyReturn(Wrapper):
    def __init__(self, env, initial_return=1, epsilon=1e-8):
        super().__init__(env)
        self.current_return = initial_return
        self.running_mean_std = RunningMeanStd()
        self.epsilon = epsilon
        self.running_mean_std.update(np.array([initial_return]))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward
        if terminated or truncated:
            self.running_mean_std.update(np.array([self.current_return]))
            self.current_return = 0
        return (
            obs,
            reward / (self.running_mean_std.mean + self.epsilon),
            terminated,
            truncated,
            info,
        )


class NormalizeRewardbyMaxReturn(Wrapper):
    def __init__(self, env, initial_return=1, max_return=None, epsilon=1e-8):
        super().__init__(env)
        self.current_return = initial_return
        self.max_return = initial_return
        self.epsilon = epsilon
        self.max_max_return = max_return

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward
        if terminated or truncated:
            self.max_return = max(self.max_return, self.current_return)
            self.current_return = 0
            if self.max_max_return is not None:
                self.max_return = min(self.max_return, self.max_max_return)
        return (
            obs,
            reward / (self.max_return + self.epsilon),
            terminated,
            truncated,
            info,
        )


def env_builder(
    domain_randomization=True,
    normalize_reward=True,
    num_frames=10,
    one_hot_action=True,
    normalize_observation=True,
    time_aware_observation=True,
    use_slow_env=False,
    last_action_wrapper=True,
    num_envs=None,
    vec_env_cls=None,
):
    if num_envs is not None:
        env = make_vec_env(
            partial(
                env_builder,
                domain_randomization=domain_randomization,
                normalize_reward=normalize_reward,
                num_frames=num_frames,
                one_hot_action=one_hot_action,
                normalize_observation=normalize_observation,
                time_aware_observation=time_aware_observation,
                use_slow_env=use_slow_env,
                last_action_wrapper=last_action_wrapper,
            ),
            n_envs=num_envs,
            vec_env_cls=vec_env_cls,
        )
        return env
    EnvClass = SlowHIVPatient if use_slow_env else FastHIVPatient
    env = EnvClass(domain_randomization=domain_randomization)
    env = TimeLimit(env, max_episode_steps=200)
    if normalize_reward:
        # env = NormalizeRewardbyReturn(env, initial_return=1e6)
        # env = NormalizeRewardbyMaxReturn(env, initial_return=1e6, max_return=1e10)
        env = TransformReward(env, lambda r: r / 1e10)
    if normalize_observation:
        env = TransformObservation(
            env, lambda obs: np.log(np.maximum(obs, 1e-10)), env.observation_space
        )
    if last_action_wrapper:
        env = LatestActionWrapper(env, one_hot_action=one_hot_action)
    if time_aware_observation:
        env = TimeAwareObservation(env)
    if num_frames > 1:
        env = FrameStackObservation(env, num_frames)
    env = Monitor(env)
    return env


class PrintMessageCallback(BaseCallback):
    def __init__(self, message: str):
        super().__init__(verbose=1)
        self.message = message

    def _on_step(self) -> bool:
        print(self.message)
        return True


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        env_name: str = None,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []
        self.env_name = env_name

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}"
            )

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            if self.env_name is not None:
                log_prefix = f"eval/{self.env_name}/"
            else:
                log_prefix = "eval/"
            self.logger.record(f"{log_prefix}mean_reward", float(mean_reward))
            self.logger.record(f"{log_prefix}mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


def create_callbacks(
    checkpoint_path,
    env_kwargs: Optional[Dict[str, Any]] = None,
    save_freq=100_000,
    eval_freq=50_000,
    num_envs=1,
    n_eval_episodes=10,
    num_eval_envs=12,
):
    env_kwargs = env_kwargs.copy() if env_kwargs is not None else None
    env_kwargs["num_envs"] = num_eval_envs
    env_kwargs["normalize_reward"] = False
    eval_freq = max(eval_freq // num_envs, 1)
    if env_kwargs is not None:
        domain_randomization = env_kwargs.pop("domain_randomization", True)
    else:
        domain_randomization = True
    dom_name = "rndenv" if domain_randomization else "detenv"
    other_dom_name = "detenv" if domain_randomization else "rndenv"
    return [
        CheckpointCallback(save_freq=save_freq, save_path=checkpoint_path),
        WandbCallback(),
        EvalCallback(
            env_builder(domain_randomization=domain_randomization, **env_kwargs),
            deterministic=True,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            callback_after_eval=PrintMessageCallback(
                "with env: " + str_of_eval(domain_randomization, False, True) + "\n" + "-" * 100
            ),
            env_name=dom_name + "_det-eval",
        ),
        EvalCallback(
            env_builder(domain_randomization=domain_randomization, **env_kwargs),
            deterministic=False,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            callback_after_eval=PrintMessageCallback(
                "with env: " + str_of_eval(domain_randomization, False, False) + "\n" + "-" * 100
            ),
            env_name=dom_name + "_stoch-eval",
        ),
        EvalCallback(
            env_builder(domain_randomization=not domain_randomization, **env_kwargs),
            deterministic=False,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            env_name=other_dom_name + "_stoch-eval",
            callback_after_eval=PrintMessageCallback(
                "with env: " + str_of_eval(not domain_randomization, False, False) + "\n" + "-" * 100
            ),
        ),
        EvalCallback(
            env_builder(domain_randomization=not domain_randomization, **env_kwargs),
            deterministic=True,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            env_name=other_dom_name + "_det-eval",
            callback_after_eval=PrintMessageCallback(
                "with env: " + str_of_eval(not domain_randomization, False, True) + "\n" + "-" * 100
            ),
        ),
    ]


@th.no_grad()
def test_model(
    model, n_eval_episodes=5, exp_name=None, slow_env=False, env_kwargs=None
):
    if env_kwargs is None:
        env_kwargs = {}
    env = env_builder(
        domain_randomization=False,
        normalize_reward=False,
        use_slow_env=slow_env,
        **env_kwargs,
    )
    print("Evaluating model without domain randomization")
    deterministic_policy_no_randomization, _ = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (no randomization): {np.mean(deterministic_policy_no_randomization):.2e} +/- {np.std(deterministic_policy_no_randomization):.2e}"
    )
    stochastic_policy_no_randomization, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
        deterministic=False,
    )
    print(
        f"Mean reward (with random sampling): {np.mean(stochastic_policy_no_randomization):.2e} +/- {np.std(stochastic_policy_no_randomization):.2e}"
    )

    rnd_env = env_builder(
        domain_randomization=True, normalize_reward=False, **env_kwargs
    )
    print("Evaluating model with domain randomization")
    deterministic_policy_with_randomization, _ = evaluate_policy(
        model, rnd_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (with randomization): {np.mean(deterministic_policy_with_randomization):.2e} +/- {np.std(deterministic_policy_with_randomization):.2e}"
    )
    stochastic_policy_with_randomization, _ = evaluate_policy(
        model,
        rnd_env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
        deterministic=False,
    )
    print(
        f"Mean reward (with random sampling): {np.mean(stochastic_policy_with_randomization):.2e} +/- {np.std(stochastic_policy_with_randomization):.2e}"
    )

    data = {
        "Deterministic env with deterministic sampling": deterministic_policy_no_randomization,
        "Random env with deterministic sampling": deterministic_policy_with_randomization,
        "Random env with random sampling": stochastic_policy_with_randomization,
        "Deterministic env with random sampling": stochastic_policy_no_randomization,
    }

    plot_scatters(exp_name, data, log_x=True)


def simulate_model(model, exp_name, domain_randomization=False, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    env = env_builder(
        domain_randomization=domain_randomization,
        normalize_reward=False,
        **env_kwargs,
    )
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    all_rewards = []
    while not done and not truncated:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        all_rewards.append(reward)
    print(f"Total reward: {total_reward:.2e}")
    plt.figure()
    plt.hist(all_rewards, bins=30)
    plt.savefig(f"plots/{exp_name}_rewards.png")
    plt.close()
