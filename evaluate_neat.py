from test_neat import evaluate_policy
from argparse import ArgumentParser
from src.common import env_builder
import pickle
import neat

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(type=str, dest="model_path")
    parser.add_argument("--n-episodes", "-e", type=int, default=20)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--det-eval", action="store_true")
    args = parser.parse_args()
    env_kwargs = dict(
        domain_randomization=args.domain_randomization,
        normalize_reward=False,
        num_frames=1,
        normalize_observation=True,
        time_aware_observation=False,
        last_action_wrapper=False,
        num_envs=args.num_envs,
    )
    env = env_builder(**env_kwargs, use_slow_env=False)
    config = neat.Config(
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.neat",
    )
    with open(args.model_path, "rb") as f:
        genome = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(genome, config)

    mean_reward, std_reward = evaluate_policy(
        winner_net, env, args.n_episodes, deterministic=args.det_eval
    )
    print(f"Mean reward: {mean_reward:.2e} ± {std_reward:.2e}")
    slow_env = env_builder(
        **env_kwargs,
        use_slow_env=True,
    )
    mean_reward_slow, std_reward_slow = evaluate_policy(
        winner_net, slow_env, args.n_episodes, deterministic=args.det_eval
    )
    print(f"Mean reward slow: {mean_reward_slow:.2e} ± {std_reward_slow:.2e}")
