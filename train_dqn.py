from gymnasium.wrappers import (
    TimeLimit,
    TransformReward,
    TransformObservation,
    FrameStackObservation,
    TimeAwareObservation,
)
# from src.env_hiv import HIVPatient
from src.env_hiv_fast import FastHIVPatient as HIVPatient
from src.train import LatestActionWrapper
from stable_baselines3 import DQN
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from coolname import generate_slug
from stable_baselines3.common.vec_env import VecNormalize
from time import time
from argparse import ArgumentParser
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial
import traceback
import torch as th
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback

ROOT_DIR = Path(__file__).parent


def env_builder(
    domain_randomization=True,
    normalize_reward=True,
    num_frames=10,
    one_hot_action=True,
    normalize_observation=True,
):
    env = HIVPatient(domain_randomization=domain_randomization)
    env = Monitor(env)
    if normalize_observation:
        env = TransformObservation(
            env, lambda obs: np.log(np.maximum(obs, 0) + 1), env.observation_space
        )
    env = LatestActionWrapper(env, one_hot_action=one_hot_action)
    env = TimeLimit(env, max_episode_steps=200)
    env = TimeAwareObservation(env)
    env = FrameStackObservation(env, num_frames)
    return env


POLICY_KWARGS = dict(
    net_arch=[256, 256],
    activation_fn=th.nn.ReLU,
)


def train_model(
    num_steps,
    exp_name,
    device="auto",
    testing=False,
    num_frames=10,
    domain_randomization=True,
    normalize_reward=True,
    one_hot_action=True,
    normalize_observation=True,
    checkpoint=None,
    learning_rate=1e-4,
):
    env = make_vec_env(
        partial(
            env_builder,
            domain_randomization=domain_randomization,
            normalize_reward=normalize_reward,
            num_frames=num_frames,
            one_hot_action=one_hot_action,
            normalize_observation=normalize_observation,
        ),
        n_envs=1,  # DQN only supports single environment
        vec_env_cls=DummyVecEnv,
    )
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_obs=False,
        clip_reward=False,
        gamma=0.99,
        epsilon=1e-8,
    )

    dqn_kwargs = dict(
        learning_rate=learning_rate,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    checkpoint_path = ROOT_DIR / "checkpoints" / exp_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=checkpoint_path,
        name_prefix="checkpoint",
        save_replay_buffer=True,
    )

    learn_kwargs = {}
    if not testing:
        wandb.init(
            project="mva-rl-assignment-Butanium", name=exp_name, sync_tensorboard=True
        )
        dqn_kwargs["tensorboard_log"] = f"logs/{exp_name}"
        learn_kwargs["callback"] = [WandbCallback(), checkpoint_callback]

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        policy_kwargs=POLICY_KWARGS,
        **dqn_kwargs,
    )

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        model = DQN.load(checkpoint, env=env, device=device)

    try:
        model.learn(total_timesteps=num_steps, progress_bar=True, **learn_kwargs)
    except Exception as e:
        print("Error during training")
        print(traceback.format_exc())
    finally:
        if not testing:
            print(f"Saving model to {exp_name}_{model.num_timesteps}")
            model.save(f"models/{exp_name}_{model.num_timesteps}")
    return model, env

@th.no_grad()
def test_model(model, n_eval_episodes=5, exp_name=None, env=None):
    if env is not None:
        original_env = env
        print("Evaluating model with original environment")
        ep_rewards_deterministic, _ = evaluate_policy(
            model, original_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
        )
        print(
            f"Mean reward (no randomization): {np.mean(ep_rewards_deterministic):.2e} +/- {np.std(ep_rewards_deterministic):.2e}"
        )
        print("evaluating with random sampling")
        ep_rewards_random, _ = evaluate_policy(
            model, original_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True, deterministic=False
        )
        print(
            f"Mean reward (with random sampling): {np.mean(ep_rewards_random):.2e} +/- {np.std(ep_rewards_random):.2e}"
        )
    env = env_builder(domain_randomization=False, normalize_reward=False)
    print("Evaluating model without domain randomization")
    ep_rewards_deterministic, _ = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (no randomization): {np.mean(ep_rewards_deterministic):.2e} +/- {np.std(ep_rewards_deterministic):.2e}"
    )

    rnd_env = env_builder(domain_randomization=True, normalize_reward=False)
    print("Evaluating model with domain randomization")
    ep_rewards_rnd_deterministic, _ = evaluate_policy(
        model, rnd_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (with randomization): {np.mean(ep_rewards_rnd_deterministic):.2e} +/- {np.std(ep_rewards_rnd_deterministic):.2e}"
    )

    # Create visualization
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ep_rewards_deterministic,
            y=np.zeros_like(ep_rewards_deterministic),
            mode="markers",
            name="No randomization",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ep_rewards_rnd_deterministic,
            y=np.ones_like(ep_rewards_rnd_deterministic),
            mode="markers",
            name="With randomization",
        )
    )
    fig.update_layout(showlegend=True)
    fig.write_html(f"plots/{exp_name}.html")


def simulate_model(model, exp_name):
    env = env_builder(domain_randomization=False, normalize_reward=False)
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
    print(f"Total reward: {total_reward}")
    plt.figure()
    plt.hist(all_rewards, bins=30)
    plt.savefig(f"plots/{exp_name}_rewards.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--no-domain-randomization",
        action="store_false",
        help="Disable domain randomization",
        dest="domain_randomization",
    )
    parser.add_argument(
        "--no-normalize-reward", action="store_false", dest="normalize_reward"
    )
    parser.add_argument(
        "--no-normalize-observation", action="store_false", dest="normalize_observation"
    )
    parser.add_argument(
        "--no-one-hot-action", action="store_false", dest="one_hot_action"
    )
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1_000_000)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--name", type=str, default=generate_slug(2))
    args = parser.parse_args()
    print(f"using {args}")
    
    model, env = train_model(
        args.num_steps,
        "dqn_mlp_" + str(int(time())) + "_" + args.name,
        device=args.device,
        testing=args.testing,
        num_frames=args.num_frames,
        domain_randomization=args.domain_randomization,
        normalize_reward=args.normalize_reward,
        one_hot_action=args.one_hot_action,
        normalize_observation=args.normalize_observation,
        checkpoint=args.checkpoint,
        learning_rate=args.learning_rate,
    )
    test_model(model, args.n_eval_episodes, args.exp_name, env=env)
    simulate_model(model, args.exp_name) 