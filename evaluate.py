from gymnasium.wrappers import (
    TimeLimit,
    TransformObservation,
    FrameStackObservation,
    TimeAwareObservation,
)
from src.env_hiv import SlowHIVPatient as HIVPatient
from src.train import LatestActionWrapper
from stable_baselines3 import DQN, PPO
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def env_builder(
    domain_randomization=True,
    normalize_reward=True,
    num_frames=10,
    one_hot_action=True,
    normalize_observation=True,
):
    env = HIVPatient(domain_randomization=domain_randomization)
    if normalize_observation:
        env = TransformObservation(
            env, lambda obs: np.log(np.maximum(obs, 0) + 1), env.observation_space
        )
    env = LatestActionWrapper(env, one_hot_action=one_hot_action)
    env = TimeLimit(env, max_episode_steps=200)
    env = TimeAwareObservation(env)
    env = FrameStackObservation(env, num_frames)
    return env


@torch.no_grad()
def evaluate_model(
    model_path,
    n_eval_episodes=10,
    domain_randomization=False,
    normalize_observation=True,
    one_hot_action=True,
    num_frames=10,
    device="auto",
    model_type="dqn",
):
    # Create environment
    env = env_builder(
        domain_randomization=domain_randomization,
        normalize_reward=False,
        num_frames=num_frames,
        one_hot_action=one_hot_action,
        normalize_observation=normalize_observation,
    )
    env = DummyVecEnv([lambda: env])

    # Load model
    model_class = DQN if model_type.lower() == "dqn" else PPO
    model = model_class.load(model_path, env=env, device=device)

    # Evaluate with deterministic actions
    print("\nEvaluating with deterministic actions:")
    rewards_deterministic, _ = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    print(
        f"Mean reward: {np.mean(rewards_deterministic):.2e} +/- {np.std(rewards_deterministic):.2e}"
    )
    print(f"Min reward: {np.min(rewards_deterministic):.2e}")
    print(f"Max reward: {np.max(rewards_deterministic):.2e}")

    # Evaluate with stochastic actions
    print("\nEvaluating with stochastic actions:")
    rewards_stochastic, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
        deterministic=False,
    )
    print(
        f"Mean reward: {np.mean(rewards_stochastic):.2e} +/- {np.std(rewards_stochastic):.2e}"
    )
    print(f"Min reward: {np.min(rewards_stochastic):.2e}")
    print(f"Max reward: {np.max(rewards_stochastic):.2e}")

    # Create visualization
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=rewards_deterministic,
            name="Deterministic",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        )
    )
    fig.add_trace(
        go.Box(
            y=rewards_stochastic,
            name="Stochastic",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        )
    )
    fig.update_layout(
        title="Episode Rewards Distribution",
        yaxis_title="Episode Reward",
        showlegend=True,
    )

    # Save plots
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    model_name = Path(model_path).stem
    fig.write_html(output_dir / f"{model_name}_rewards_distribution.html")
    
    # Save reward statistics
    stats = {
        "deterministic": {
            "mean": np.mean(rewards_deterministic),
            "std": np.std(rewards_deterministic),
            "min": np.min(rewards_deterministic),
            "max": np.max(rewards_deterministic),
        },
        "stochastic": {
            "mean": np.mean(rewards_stochastic),
            "std": np.std(rewards_stochastic),
            "min": np.min(rewards_stochastic),
            "max": np.max(rewards_stochastic),
        },
    }
    
    return stats


def simulate_episode(
    model_path,
    domain_randomization=False,
    normalize_observation=True,
    one_hot_action=True,
    num_frames=10,
    device="auto",
    model_type="dqn",
    deterministic=True,
):
    env = env_builder(
        domain_randomization=domain_randomization,
        normalize_reward=False,
        num_frames=num_frames,
        one_hot_action=one_hot_action,
        normalize_observation=normalize_observation,
    )
    
    model_class = DQN if model_type.lower() == "dqn" else PPO
    model = model_class.load(model_path, device=device)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    all_rewards = []
    
    while not done and not truncated:
        action = model.predict(obs, deterministic=deterministic)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        all_rewards.append(reward)
    
    print(f"\nSimulation Results:")
    print(f"Total reward: {total_reward:.2e}")
    print(f"Episode length: {len(all_rewards)}")
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_rewards, bins=30, alpha=0.7)
    plt.title("Reward Distribution During Episode")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    model_name = Path(model_path).stem
    plt.savefig(output_dir / f"{model_name}_episode_rewards.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["dqn", "ppo"],
        default="dqn",
        help="Type of the model (DQN or PPO)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Enable domain randomization during evaluation",
    )
    parser.add_argument(
        "--no-normalize-observation",
        action="store_false",
        dest="normalize_observation",
    )
    parser.add_argument(
        "--no-one-hot-action",
        action="store_false",
        dest="one_hot_action",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Evaluate the model
    stats = evaluate_model(
        model_path=args.model_path,
        n_eval_episodes=args.n_eval_episodes,
        domain_randomization=args.domain_randomization,
        normalize_observation=args.normalize_observation,
        one_hot_action=args.one_hot_action,
        num_frames=args.num_frames,
        device=args.device,
        model_type=args.model_type,
    )

    # Simulate a single episode
    simulate_episode(
        model_path=args.model_path,
        domain_randomization=args.domain_randomization,
        normalize_observation=args.normalize_observation,
        one_hot_action=args.one_hot_action,
        num_frames=args.num_frames,
        device=args.device,
        model_type=args.model_type,
    ) 