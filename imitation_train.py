import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from imitation.algorithms import bc
from imitation.data import serialize
from imitation.data.rollout import (
    TrajectoryAccumulator,
    types,
    GenTrajTerminationFn,
    make_sample_until,
    spaces,
    rollout_stats,
    unwrap_traj,
    dataclasses,
)
from imitation.util.logger import configure as configure_logger
from imitation.util.util import save_policy
from imitation.data import rollout
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformObservation
from coolname import generate_slug
from tqdm.rich import trange, tqdm
import wandb

from src.env_hiv_fast import FastHIVPatient


@dataclass
class DoOnce:
    do: bool = True

    def __call__(self):
        if self.do:
            self.do = False
            return True
        return False


def build_env(domain_randomization: bool):
    env = FastHIVPatient(domain_randomization=domain_randomization)
    env = TransformObservation(
        env,
        lambda obs: np.log(np.maximum(obs, 1e-8)),
        env.observation_space,
    )
    env = TimeLimit(env, max_episode_steps=200)
    return env


def generate_heuristic_rollouts(
    add_fixed_env: bool,
    n_envs: int,
    num_rollouts: int,
    rng: np.random.Generator,
) -> types.TrajectoryWithRew:
    all_trajectories = []

    for _ in trange(0, num_rollouts, n_envs, desc="Generating rollouts"):
        venv = make_vec_env(
            lambda *, _do_once: build_env(_do_once()),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=dict(_do_once=DoOnce(add_fixed_env)),
        )
        sample_until = make_sample_until(min_episodes=n_envs)
        # Collect rollout tuples.
        # accumulator for incomplete trajectories
        trajectories_accum = TrajectoryAccumulator()
        trajectories = []
        obs = venv.reset()
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # we use dictobs to iterate over the envs in a vecenv
        for env_idx, ob in enumerate(wrapped_obs):
            # Seed with first obs only. Inside loop, we'll only add second obs from
            # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
            # get all observations, but they're not duplicated into "next obs" and
            # "previous obs" (this matters for, e.g., Atari, where observations are
            # really big).
            trajectories_accum.add_step(dict(obs=ob), env_idx)

        # Now, we sample until `sample_until(trajectories)` is true.
        # If we just stopped then this would introduce a bias towards shorter episodes,
        # since longer episodes are more likely to still be active, i.e. in the process
        # of being sampled from. To avoid this, we continue sampling until all epsiodes
        # are complete.
        #
        # To start with, all environments are active.
        active = np.ones(venv.num_envs, dtype=bool)
        dones = np.zeros(venv.num_envs, dtype=bool)
        while np.any(active):
            # policy gets unwrapped observations (eg as dict, not dictobs)
            acts = np.array(
                venv.env_method(
                    "greedy_action", num_watch_steps=5, consecutive_actions=1
                )
            )
            obs, rews, dones, infos = venv.step(acts)
            assert isinstance(
                obs,
                (np.ndarray, dict),
            ), "Tuple observations are not supported."
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            # If an environment is inactive, i.e. the episode completed for that
            # environment after `sample_until(trajectories)` was true, then we do
            # *not* want to add any subsequent trajectories from it. We avoid this
            # by just making it never done.
            dones &= active

            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                acts,
                wrapped_obs,
                rews,
                dones,
                infos,
            )
            trajectories.extend(new_trajs)

            if sample_until(trajectories):
                # Termination condition has been reached. Mark as inactive any
                # environments where a trajectory was completed this timestep.
                active &= ~dones

        all_trajectories.extend(trajectories)
    trajectories = all_trajectories

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        if isinstance(venv.observation_space, spaces.Dict):
            exp_obs = {}
            for k, v in venv.observation_space.items():
                assert v.shape is not None
                exp_obs[k] = (n_steps + 1,) + v.shape
        else:
            obs_space_shape = venv.observation_space.shape
            assert obs_space_shape is not None
            exp_obs = (n_steps + 1,) + obs_space_shape  # type: ignore[assignment]
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        assert venv.action_space.shape is not None
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    # trajectories = [unwrap_traj(traj) for traj in trajectories]
    trajectories = [dataclasses.replace(traj, infos=None) for traj in trajectories]
    stats = rollout_stats(trajectories)
    print(f"Rollout stats: {stats}")
    return trajectories


def main(
    num_rollouts: int,
    num_envs: int,
    exp_name: str,
    add_fixed_env: bool = True,
    device: str = "auto",
    rollout_path: Path | None = None,
    n_epochs: int = 5,
):
    rng = np.random.default_rng()
    if rollout_path is None:
        rollouts = generate_heuristic_rollouts(
            add_fixed_env, num_envs, num_rollouts, rng=rng
        )
        save_path = Path("data/rollouts") / (
            exp_name + "_" + str(num_rollouts) + ".traj"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serialize.save(save_path, rollouts)
        print(f"Saved rollouts to {save_path}")
    else:
        rollouts = serialize.load(rollout_path)
        print(f"Loaded rollouts from {rollout_path}")
    transitions = rollout.flatten_trajectories(rollouts)
    env = make_vec_env(
        lambda: build_env(domain_randomization=True),
        n_envs=num_envs,
    )
    det_env = make_vec_env(
        lambda: build_env(domain_randomization=False),
        n_envs=num_envs,
    )
    dummy_env = FastHIVPatient(domain_randomization=False)
    bc_trainer = bc.BC(
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        demonstrations=transitions,
        rng=rng,
        device=device,
        custom_logger=configure_logger(
            folder=Path("logs") / exp_name, format_strs=["wandb", "log"]
        ),
    )
    bc_trainer.train(n_epochs=n_epochs)
    mean_reward, std_reward = evaluate_policy(
        bc_trainer.policy, env, n_eval_episodes=10
    )
    det_mean_reward, det_std_reward = evaluate_policy(
        bc_trainer.policy, det_env, n_eval_episodes=10
    )
    print(f"Reward: {mean_reward:.2e} ± {std_reward:.2e}")
    print(f"Det env reward: {det_mean_reward:.2e} ± {det_std_reward:.2e}")
    # save policy
    save_path = Path("models/bc") / (exp_name + "_" + str(num_rollouts) + ".pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_policy(bc_trainer.policy, save_path)
    # bc_trainer.save_policy(save_path)
    print(f"Saved policy to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--no-fixed-env", action="store_false", dest="add_fixed_env")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rollout-path", "-p", type=Path, default=None)
    parser.add_argument("--n-epochs", type=int, default=5)
    args = parser.parse_args()
    exp_name = str(int(time.time())) + "_" + generate_slug(2)
    print(f"Experiment name: {exp_name}")
    wandb.init(project="hiv-imitation", name=exp_name, sync_tensorboard=True)
    main(
        num_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        exp_name=exp_name,
        add_fixed_env=args.add_fixed_env,
        device=args.device,
        rollout_path=args.rollout_path,
        n_epochs=args.n_epochs,
    )
