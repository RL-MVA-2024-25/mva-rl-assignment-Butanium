from gymnasium.wrappers import (
    TimeLimit,
    TransformReward,
    TransformObservation,
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
import torch as th


def env_builder(
    domain_randomization=True,
    normalize_reward=True,
    num_frames=10,
    one_hot_action=True,
    normalize_observation=True,
):
    env = HIVPatient(domain_randomization=domain_randomization)
    env = Monitor(env)
    if normalize_reward:
        env = TransformReward(env, lambda reward: reward / 50000.0)
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
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=th.nn.ReLU,
    ortho_init=False,
    log_std_init=-2.0,
)


def train_model(
    num_steps,
    exp_name,
    device="auto",
    testing=False,
    num_envs=5,
    num_frames=10,
    domain_randomization=True,
    normalize_reward=True,
    one_hot_action=True,
    normalize_observation=True,
    checkpoint=None,
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
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv if num_envs > 1 else None,
    )
    ppo_kwargs = dict(
        batch_size=128,
        n_steps=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        # use_sde=True,
        # sde_sample_freq=4,
    )
    learn_kwargs = {}
    if not testing:
        wandb.init(
            project="mva-rl-assignment-Butanium", name=exp_name, sync_tensorboard=True
        )
        ppo_kwargs["tensorboard_log"] = f"logs/{exp_name}"
        learn_kwargs["callback"] = WandbCallback()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        policy_kwargs=POLICY_KWARGS,
        **ppo_kwargs,
    )
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
    fig.add_trace(
        go.Scatter(
            x=ep_rewards_not_deterministic,
            y=2 * np.ones_like(ep_rewards_not_deterministic),
            mode="markers",
            name="No randomization not deterministic",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ep_rewards_rnd_not_deterministic,
            y=3 * np.ones_like(ep_rewards_rnd_not_deterministic),
            mode="markers",
            name="With randomization not deterministic",
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
        "--no-domain-randomization",
        action="store_false",
        help="Disable domain randomization",
        dest="domain_randomization",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="ppo_mlp_randomized_" + str(int(time())) + "_" + generate_slug(2),
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
        one_hot_action=args.one_hot_action,
        normalize_observation=args.normalize_observation,
        checkpoint=args.checkpoint,
    )
    test_model(model, args.n_eval_episodes, args.exp_name)
    simulate_model(model, args.exp_name)
