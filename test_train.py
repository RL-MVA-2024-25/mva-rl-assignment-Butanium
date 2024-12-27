from gymnasium.wrappers import (
    TimeLimit,
    TransformReward,
    FrameStackObservation,
    TimeAwareObservation,
)
from src.env_hiv import HIVPatient
from src.train import LatestActionWrapper
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from coolname import generate_slug
from time import time
from argparse import ArgumentParser
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from functools import partial
import traceback

def env_builder(domain_randomization=True, normalize_reward=True, num_frames=10):
    env = HIVPatient(domain_randomization=domain_randomization)
    env = Monitor(env)
    if normalize_reward:
        env = TransformReward(env, lambda reward: reward / 50000.0)
    env = LatestActionWrapper(env)
    env = TimeLimit(env, max_episode_steps=200)
    env = TimeAwareObservation(env)
    env = FrameStackObservation(env, num_frames)
    return env


def train_model(
    num_steps,
    exp_name,
    device="auto",
    testing=False,
    num_envs=5,
    num_frames=10,
    domain_randomization=True,
    normalize_reward=True,
    checkpoint=None,
):
    env = make_vec_env(
        partial(
            env_builder,
            domain_randomization=domain_randomization,
            normalize_reward=normalize_reward,
            num_frames=num_frames,
        ),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
    )
    ppo_kwargs = dict(batch_size=64)
    learn_kwargs = {}
    if not testing:
        wandb.init(
            project="mva-rl-assignment-Butanium", name=exp_name, sync_tensorboard=True
        )
        ppo_kwargs["tensorboard_log"] = f"logs/{exp_name}"
        learn_kwargs["callback"] = WandbCallback()
    model = PPO("MlpPolicy", env, verbose=1, device=device, **ppo_kwargs)
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        model = PPO.load(checkpoint, env=env, device=device)
    try:
        model.learn(total_timesteps=num_steps, progress_bar=True, **learn_kwargs)
    except Exception as e:
        print("Error during training")
        # print traceback
        print(traceback.format_exc())
    finally:
        if not testing:
            print(f"Saving model to {exp_name}_{model.num_timesteps}")
            model.save(f"models/{exp_name}_{model.num_timesteps}")
    return model


def test_model(model, n_eval_episodes=5, exp_name=None):
    env = env_builder(domain_randomization=False, normalize_reward=False)
    ep_rewards_deterministic, _ = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (no randomization): {np.mean(ep_rewards_deterministic):.2e} +/- {np.std(ep_rewards_deterministic):.2e}"
    )

    rnd_env = env_builder(domain_randomization=True, normalize_reward=False)
    ep_rewards_rnd_deterministic, _ = evaluate_policy(
        model, rnd_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward (with randomization): {np.mean(ep_rewards_rnd_deterministic):.2e} +/- {np.std(ep_rewards_rnd_deterministic):.2e}"
    )

    ep_rewards_not_deterministic, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
        deterministic=False,
    )
    print(
        f"Mean reward (no randomization) not deterministic: {np.mean(ep_rewards_not_deterministic):.2e} +/- {np.std(ep_rewards_not_deterministic):.2e}"
    )

    ep_rewards_rnd_not_deterministic, _ = evaluate_policy(
        model,
        rnd_env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
        deterministic=False,
    )
    print(
        f"Mean reward (with randomization) not deterministic: {np.mean(ep_rewards_rnd_not_deterministic):.2e} +/- {np.std(ep_rewards_rnd_not_deterministic):.2e}"
    )
    # 2x2 subplots with histograms of the episode rewards
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
    fig.add_trace(
        go.Histogram(x=ep_rewards_deterministic, name="No randomization"), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=ep_rewards_rnd_deterministic, name="With randomization"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=ep_rewards_not_deterministic, name="No randomization not deterministic"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=ep_rewards_rnd_not_deterministic,
            name="With randomization not deterministic",
        ),
        row=2,
        col=2,
    )
    fig.write_html(f"plots/{exp_name}.html")


def simulate_model(model, exp_name):
    env = env_builder(domain_randomization=False, normalize_reward=False)
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    all_rewards = []
    while not done and not truncated:
        action = model.predict(obs, deterministic=False)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        all_rewards.append(reward)
    print(f"Total reward: {total_reward}")
    plt.hist(
        all_rewards, bins=30
    )  # Increased the number of bins for better granularity
    plt.savefig(f"plots/{exp_name}_rewards.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--domain-randomization",
        action="store_false",
        help="Disable domain randomization",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="ppo_mlp_randomized_" + str(int(time())) + "_" + generate_slug(2),
    )
    parser.add_argument("--normalize-reward", type=bool, default=True)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1_000_000)
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    print(f"using {args}")
    model = train_model(
        args.num_steps,
        args.exp_name,
        device=args.device,
        testing=args.testing,
        num_envs=args.num_envs,
        num_frames=args.num_frames,
        domain_randomization=args.domain_randomization,
        normalize_reward=args.normalize_reward,
        checkpoint=args.checkpoint,
    )
    test_model(model, args.n_eval_episodes, args.exp_name)
    simulate_model(model, args.exp_name)
