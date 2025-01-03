import signal

from stable_baselines3 import DQN
import wandb
from coolname import generate_slug
from time import time
from argparse import ArgumentParser
import traceback
import torch as th
from pathlib import Path
from src.common import env_builder, test_model, simulate_model, create_callbacks

ROOT_DIR = Path(__file__).parent


POLICY_KWARGS = dict(
    net_arch=[256, 256],
    activation_fn=th.nn.ReLU,
)


def train_model(
    num_steps,
    exp_name,
    device="auto",
    testing=False,
    checkpoint=None,
    learning_rate=1e-4,
    env_kwargs=None,
):
    if env_kwargs is None:
        env_kwargs = {}
    env = env_builder(**env_kwargs, num_envs=1)

    dqn_kwargs = dict(
        learning_rate=learning_rate,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.9999,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )

    checkpoint_path = ROOT_DIR / "checkpoints" / exp_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    learn_kwargs = {}
    if not testing:
        wandb.init(
            project="mva-rl-assignment-Butanium", name=exp_name, sync_tensorboard=True
        )
        dqn_kwargs["tensorboard_log"] = f"logs/{exp_name}"
        learn_kwargs["callback"] = create_callbacks(
            checkpoint_path, env_kwargs=env_kwargs, eval_freq=20_000
        )
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

    def handle_exit(signum, frame):
        print("Signal received, exiting...")
        print(f"Saving model to {exp_name}_{model.num_timesteps}")
        model.save(f"models/{exp_name}_{model.num_timesteps}")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        model.learn(
            total_timesteps=num_steps,
            progress_bar=True,
            log_interval=40,
            **learn_kwargs,
        )
    except Exception as e:
        print("Error during training")
        print(traceback.format_exc())
    finally:
        if not testing:
            print(f"Saving model to {exp_name}_{model.num_timesteps}")
            model.save(f"models/{exp_name}_{model.num_timesteps}")
    return model, env


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
    env_kwargs = dict(
        domain_randomization=args.domain_randomization,
        normalize_reward=args.normalize_reward,
        num_frames=args.num_frames,
        one_hot_action=args.one_hot_action,
        normalize_observation=args.normalize_observation,
    )
    exp_name = "dqn_mlp_" + str(int(time())) + "_" + args.name
    model, env = train_model(
        args.num_steps,
        exp_name,
        device=args.device,
        testing=args.testing,
        env_kwargs=env_kwargs,
        checkpoint=args.checkpoint,
        learning_rate=args.learning_rate,
    )
    env_kwargs.pop("domain_randomization")
    env_kwargs.pop("normalize_reward")
    print("Testing model with fast environment")
    test_model(
        model,
        args.n_eval_episodes,
        exp_name,
        env_kwargs=env_kwargs,
    )
    print("-" * 100 + "\n")
    print("Testing model with slow environment")
    test_model(
        model,
        args.n_eval_episodes,
        exp_name,
        slow_env=True,
        env_kwargs=env_kwargs,
    )
    print("-" * 100 + "\n")
    print("Simulating model...")
    simulate_model(
        model,
        exp_name,
        domain_randomization=args.domain_randomization,
        env_kwargs=env_kwargs,
    )
