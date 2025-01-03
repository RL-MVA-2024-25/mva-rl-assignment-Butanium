from functools import partial
from typing import Union, Tuple, List, Optional, Callable, Dict, Any
import warnings
from argparse import ArgumentParser
import time
from pathlib import Path

import neat
from neat.reporting import BaseReporter
from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)
import wandb
import pickle
from coolname import generate_slug
from tqdm.rich import tqdm
from tqdm import tqdm as tqdm_base
from scipy.special import softmax
from joblib import Parallel, delayed


from src.common import env_builder

import gzip
import random
import time

try:
    import cPickle as pickle  # pylint: disable=import-error
except ImportError:
    import pickle  # pylint: disable=import-error

from neat.population import Population
from neat.reporting import BaseReporter


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(
        self,
        generation_interval=100,
        time_interval_seconds=300,
        filename_prefix="neat-checkpoint-",
    ):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(
                config, population, species_set, self.current_generation
            )
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """Save the current simulation state."""
        filename = self.filename_prefix / f"{generation}.pkl"
        print("Saving checkpoint to {0}".format(filename))
        print(species_set)
        with gzip.open(filename, "wb", compresslevel=5) as f:
            data = (
                generation,
                config,
                population,
                species_set.species,
                species_set.genome_to_species,
                species_set.indexer,
                random.getstate(),
            )  # config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved checkpoint to {filename}")

    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            (
                generation,
                config,
                population,
                species,
                genome_to_species,
                indexer,
                rndstate,
            ) = pickle.load(f)
            species_set = neat.DefaultSpeciesSet(config)
            species_set.species = species
            species_set.genome_to_species = genome_to_species
            species_set.indexer = indexer
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))


class WandbReporter(BaseReporter):
    """Reports training progress to Weights & Biases."""

    def __init__(self):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        """Log start of new generation."""
        self.generation = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        """Log generation statistics."""
        ng = len(population)
        ns = len(species_set.species)

        # Calculate species stats
        species_stats = {}
        sizes = []
        fitnesses = []
        adjusted_fitnesses = []
        stagnations = []
        for species in species_set.species.values():
            sizes.append(len(species.members))
            if species.fitness is not None:
                fitnesses.append(species.fitness)
            if species.adjusted_fitness is not None:
                adjusted_fitnesses.append(species.adjusted_fitness)
            stagnations.append(self.generation - species.last_improved)

        species_stats["species/mean_size"] = mean(sizes)
        species_stats["species/mean_fitness"] = mean(fitnesses) if fitnesses else None
        species_stats["species/mean_adjusted_fitness"] = (
            mean(adjusted_fitnesses) if adjusted_fitnesses else None
        )
        species_stats["species/mean_stagnation"] = mean(stagnations)
        species_stats["species/num_species"] = ns
        # Log timing info
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)

        # Log to wandb
        self.log(
            {
                "population_size": ng,
                "num_species": ns,
                "generation_time": elapsed,
                "generation_time_avg": average,
                "num_extinctions": self.num_extinctions,
                **species_stats,
            }
        )

    def post_evaluate(self, config, population, species, best_genome):
        """Log fitness statistics."""
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)

        self.log(
            {
                "fitness/mean": fit_mean,
                "fitness/std": fit_std,
                "fitness/best": best_genome.fitness,
                "best/species_id": best_species_id,
                "best/genome_size": sum(best_genome.size()),
            }
        )

    def info(self, msg):
        print(msg)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs, step=self.generation)


class ProgressReporter(BaseReporter):
    def __init__(self, num_generations):
        self.num_generations = num_generations
        self.pbar = tqdm(total=num_generations, desc="Generation")

    def end_generation(self, config, population, species_set):
        self.pbar.update(1)


def evaluate_policy(
    net,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    pbar = tqdm(total=n_eval_episodes, desc="Evaluating")
    while (episode_counts < episode_count_targets).any():
        logits = np.stack([np.array(net.activate(obs)) for obs in observations])
        if deterministic:
            actions = np.argmax(logits, axis=1)
        else:
            probs = softmax(logits, axis=1)
            actions = [
                np.random.choice(len(probs[i]), p=probs[i]) for i in range(n_envs)
            ]

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                            pbar.update(1)
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        pbar.update(1)
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward


def eval_single_genome(
    genome_tuple, config, *, env_builder, n_episodes, deterministic_eval
):
    """Evaluate a single genome"""
    g_id, genome = genome_tuple
    env = env_builder()  # Create environment for this worker
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    episode_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            logits = net.activate(obs)
            if deterministic_eval:
                action = np.argmax(logits)
            else:
                probs = softmax(logits)
                action = np.random.choice(len(probs), p=probs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    return g_id, np.log10(max(1, mean_reward))


def eval_genome(genome_list, config, *, env_builder, n_episodes, deterministic_eval):
    """Parallel genome evaluation using joblib"""
    results = Parallel(n_jobs=-4)(
        delayed(eval_single_genome)(
            genome_tuple,
            config,
            env_builder=env_builder,
            n_episodes=n_episodes,
            deterministic_eval=deterministic_eval,
        )
        for genome_tuple in genome_list
    )
    results = dict(results)

    # Update fitness values
    for g_id, genome in genome_list:
        genome.fitness = results[g_id]


def run_neat(
    env_builder,
    config,
    *,
    checkpoint_dir,
    n_episodes=20,
    n_generations=1000,
    deterministic_eval=False,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pop = neat.Population(config)

    # Add reporters to show progress
    pop.add_reporter(WandbReporter())
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(ProgressReporter(n_generations))
    pop.add_reporter(
        Checkpointer(
            generation_interval=1,
            filename_prefix=checkpoint_dir,
            time_interval_seconds=10_000,
        )
    )
    pop.add_reporter(neat.StdOutReporter(False))
    # Run evolution
    winner = pop.run(
        partial(
            eval_genome,
            env_builder=env_builder,
            n_episodes=n_episodes,
            deterministic_eval=deterministic_eval,
        ),
        n=n_generations,
    )
    return winner, stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-episodes", "-e", type=int, default=20)
    parser.add_argument("--n-generations", "-g", type=int, default=1000)
    parser.add_argument("--domain-randomization", action="store_true")
    # parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--name", type=str, default=generate_slug(2))
    parser.add_argument("--det-eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    name = (
        (
            ("rnd-env" if args.domain_randomization else "det-env")
            + "-"
            + str(int(time.time()))
            + "-"
            + args.name
        )
        if not args.debug
        else str(int(time.time())) + "-debug"
    )
    wandb.init(project="neat-hiv", name=name)
    env_kwargs = dict(
        domain_randomization=args.domain_randomization,
        normalize_reward=False,
        num_frames=1,
        normalize_observation=True,
        time_aware_observation=False,
        last_action_wrapper=False,
    )
    env_b = partial(
        env_builder,
        **env_kwargs,
        use_slow_env=False,
        # num_envs=args.num_envs,
    )
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.neat" if not args.debug else "config_debug.neat",
    )
    winner, stats = run_neat(
        env_b,
        config,
        checkpoint_dir=Path("checkpoints") / name,
        n_episodes=args.n_episodes if not args.debug else 1,
        n_generations=args.n_generations if not args.debug else 2,
        deterministic_eval=args.det_eval,
    )
    save_path = Path("models") / "neat" / (name + ".pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(winner, f)
    save_path_all = save_path.parent / ("stats_" + name + ".pkl")
    with open(save_path_all, "wb") as f:
        pickle.dump(stats, f)
    # eval winner on slow and fast env
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    mean_reward, std_reward = evaluate_policy(
        winner_net,
        env_builder(**env_kwargs, num_envs=1),
        args.n_episodes * 2,
        deterministic=args.det_eval,
    )
    print(f"winner_reward: {mean_reward:.2e} ± {std_reward:.2e}")
    slow_env = env_builder(**{**env_kwargs, "use_slow_env": True, "num_envs": 1})
    mean_reward_slow, std_reward_slow = evaluate_policy(
        winner_net, slow_env, args.n_episodes * 2, deterministic=args.det_eval
    )
    print(f"winner_reward_slow: {mean_reward_slow:.2e} ± {std_reward_slow:.2e}")
    wandb.log(
        {
            "winner_reward_slow": mean_reward_slow,
            "winner_reward_slow_std": std_reward_slow,
            "winner_reward": mean_reward,
            "winner_reward_std": std_reward,
        }
    )
    wandb.finish()
