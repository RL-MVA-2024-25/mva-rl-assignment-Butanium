from stable_baselines3 import DQN, PPO
from argparse import ArgumentParser
from src.common import test_model, simulate_model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        type=str,
        help="Path to the saved model",
        dest="model_path",
    )
    parser.add_argument(
        "--model-type",
        "-t",
        type=str,
        choices=["dqn", "ppo"],
        default="dqn",
        help="Type of the model (DQN or PPO)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=10)
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
    exp_name = "eval_" + args.model_path.split("/")[-1]
    model_class = DQN if args.model_type.lower() == "dqn" else PPO
    model = model_class.load(args.model_path, device=args.device)
    env_kwargs = dict(
        num_frames=args.num_frames,
        one_hot_action=args.one_hot_action,
        normalize_observation=args.normalize_observation,
    )
    # Evaluate the model
    print("Evaluating model on fast environment")
    test_model(
        model,
        args.n_eval_episodes,
        exp_name + "_fast",
        slow_env=False,
        env_kwargs=env_kwargs,
    )
    print("Evaluating model on slow environment")
    test_model(
        model,
        args.n_eval_episodes,
        exp_name + "_slow",
        slow_env=True,
        env_kwargs=env_kwargs,
    )
    print("Simulating model on fast deterministic environment")
    simulate_model(
        model,
        exp_name + "_fast_deterministic",
        domain_randomization=False,
        env_kwargs=env_kwargs,
    )
    print("Simulating model on fast random environment")
    simulate_model(
        model,
        exp_name + "_fast_random",
        domain_randomization=True,
        env_kwargs=env_kwargs,
    )
