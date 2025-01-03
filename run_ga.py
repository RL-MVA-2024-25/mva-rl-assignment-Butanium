from functools import partial
import numpy as np
import gymnasium as gym
from pathlib import Path
import wandb
from tqdm.rich import tqdm
import time
from typing import Callable
import pickle
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
from coolname import generate_slug
import json
from src.common import env_builder


def debug(*args, **kwargs):
    print(*args, **kwargs)

MCTS_CHAMPION = [3, 3, 0, 2, 1, 3, 1, 3, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 1, 0, 0, 0, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 1, 1, 2, 0, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
champion_paths =  ["/dlabscratch1/cdumas/mva-rl-assignment-Butanium/models/ga/ga-det-1735824509-zero-rollout-200pop-normal-mut-ls", "/dlabscratch1/cdumas/mva-rl-assignment-Butanium/models/ga/ga-det-1735825461-random-rollout-200pop-normal-mut-ls", "/dlabscratch1/cdumas/mva-rl-assignment-Butanium/models/ga/ga-det-1735827405-z-rollout-300pop-normal-mut-ls-fsr0.6", "/dlabscratch1/cdumas/mva-rl-assignment-Butanium/models/ga/ga-det-1735827648-3-rollout-300pop-normal-mut-ls-fsr0.6"]

ga_champions = []
for path in champion_paths:
    with open(Path(path) / "ga_solution.json", "r") as f:
        ga_champions.append(json.load(f)["best_solution"])

CHAMPIONS = np.array(ga_champions + [MCTS_CHAMPION])


@dataclass
class GAConfig:
    num_generations: int = 1000
    population_size: int = 50
    mutation_prob: float | np.ndarray = 0.1
    sequence_length: int = 200
    num_actions: int = 4
    use_parallel: bool = True
    num_jobs: int = -4
    selection_strategy: str = "top_n"
    parallel_mutation: bool = False
    initial_selection_rate: float = 1.0
    final_selection_rate: float = 0.1
    num_elites: int = 4
    max_stagnation: int = -1
    end_rollout_fn: Callable | None = None

    def __post_init__(self):
        if isinstance(self.mutation_prob, float):
            self.mutation_prob = np.full(self.sequence_length, self.mutation_prob)


class GeneticAlgorithm:
    def __init__(
        self,
        config: GAConfig,
        env_builder: Callable,
        initial_population: Optional[np.ndarray] = None,

    ):
        self.config = config
        self.env_builder = env_builder
        self.population = (
            self._initialize_population()
            if initial_population is None
            else initial_population
        )
        self.best_fitness = float("-inf")
        self.best_solution = None
        self.generation = 0
        self.last_generation_fitness = []
        self.step_times = {}
        self._selection_strategies = {
            "tournament": self._tournament_selection,
            "top_n": self._top_n_selection,
            "roulette": self._roulette_selection,
        }
        self.current_selection_rate = config.initial_selection_rate
        self.generations_without_improvement = 0
        self.should_stop = False
        self.stats = {}

    def _initialize_population(self) -> np.ndarray:
        return np.random.randint(
            0,
            self.config.num_actions,
            size=(self.config.population_size, self.config.sequence_length),
        )

    def _initialize_population_from_individual(
        self, individual: np.ndarray
    ) -> np.ndarray:
        return self._mutate(np.array([individual] * self.config.population_size))
        

    def _evaluate_solution(self, solution: np.ndarray) -> float:
        env = self.env_builder()
        obs, _ = env.reset()
        total_reward = 0
        i = 0
        while True:
            if i < self.config.sequence_length:
                action = solution[i]
            else:
                if self.config.end_rollout_fn is None:
                    break
                action = self.config.end_rollout_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break

        return np.log10(max(1, total_reward))

    def _evaluate_population(self) -> np.ndarray:
        start_time = time.time()
        if self.config.use_parallel:
            fitness_scores = Parallel(n_jobs=self.config.num_jobs)(
                delayed(self._evaluate_solution_wrapper)(solution, self.env_builder)
                for solution in self.population
            )
        else:
            fitness_scores = [
                self._evaluate_solution(solution) for solution in self.population
            ]
        self.step_times["evaluate"] = time.time() - start_time
        return np.array(fitness_scores)

    @staticmethod
    def _evaluate_solution_wrapper(
        solution: np.ndarray, env_builder: Callable
    ) -> float:
        # Static method wrapper for parallel processing
        env = env_builder()
        obs, _ = env.reset()
        total_reward = 0

        for action in solution:
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            if terminated or truncated:
                break

        return np.log10(max(1, total_reward))

    def _get_current_selection_rate(self) -> float:
        # Linear decay from initial to final selection rate
        progress = self.generation / (self.config.num_generations - 1)
        return (
            self.config.initial_selection_rate
            + (self.config.final_selection_rate - self.config.initial_selection_rate)
            * progress
        )

    def _select_parents(self, fitness_scores: np.ndarray) -> np.ndarray:
        start_time = time.time()
        selection_func = self._selection_strategies.get(self.config.selection_strategy)
        if selection_func is None:
            raise ValueError(
                f"Unknown selection strategy: {self.config.selection_strategy}"
            )

        # Update current selection rate
        self.current_selection_rate = self._get_current_selection_rate()

        # Calculate number of parents based on current selection rate
        num_parents = int(self.config.population_size * self.current_selection_rate)
        selected_parents = selection_func(fitness_scores, num_parents)

        self.step_times["select"] = time.time() - start_time
        return selected_parents

    def _tournament_selection(
        self, fitness_scores: np.ndarray, num_parents: int
    ) -> np.ndarray:
        selected_parents = []
        for _ in range(num_parents):
            tournament_idx = np.random.choice(
                len(self.population), size=3, replace=False
            )
            tournament_fitness = fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected_parents.append(self.population[winner_idx])
        return np.array(selected_parents)

    def _top_n_selection(
        self, fitness_scores: np.ndarray, num_parents: int
    ) -> np.ndarray:
        top_indices = np.argsort(fitness_scores)[-num_parents:]
        return self.population[top_indices]

    def _roulette_selection(
        self, fitness_scores: np.ndarray, num_parents: int
    ) -> np.ndarray:
        # Shift fitness scores to be positive
        shifted_fitness = fitness_scores - np.min(fitness_scores) + 1e-6
        probabilities = shifted_fitness / shifted_fitness.sum()
        selected_indices = np.random.choice(
            len(self.population),
            size=num_parents,
            p=probabilities,
            replace=False,
        )
        return self.population[selected_indices]

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        start_time = time.time()
        offspring = []
        num_offspring = self.config.population_size - self.config.num_elites

        while len(offspring) < num_offspring:
            parent1, parent2 = random.sample(list(parents), k=2)
            # Single-point crossover
            crossover_point = random.randint(1, self.config.sequence_length - 1)
            child = np.concatenate(
                [parent1[:crossover_point], parent2[crossover_point:]]
            )
            offspring.append(child)

        self.step_times["crossover"] = time.time() - start_time
        return np.array(offspring)

    def _mutate(self, solutions: np.ndarray) -> np.ndarray:
        start_time = time.time()
        if self.config.use_parallel and self.config.parallel_mutation:
            solutions = np.array(
                Parallel(n_jobs=self.config.num_jobs)(
                    delayed(self._mutate_single)(
                        solution, self.config.mutation_prob, self.config.num_actions
                    )
                    for solution in solutions
                )
            )
        else:
            for solution in solutions:
                self._mutate_single(
                    solution, self.config.mutation_prob, self.config.num_actions
                )
        self.step_times["mutate"] = time.time() - start_time
        return solutions

    @staticmethod
    def _mutate_single(
        solution: np.ndarray, mutation_prob: np.ndarray, num_actions: int
    ) -> np.ndarray:
        # Static method for parallel mutation
        solution = solution.copy()
        for idx in range(len(solution)):
            if random.random() < mutation_prob[idx]:
                solution[idx] = random.randint(0, num_actions - 1)
        return solution

    def step(self) -> Tuple[float, float, float]:
        self.stats = {}
        # Evaluate current population
        fitness_scores = self._evaluate_population()
        self.last_generation_fitness = fitness_scores

        # Update best solution
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_solution = self.population[best_idx].copy()
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        if (
            self.generations_without_improvement >= self.config.max_stagnation
            and self.config.max_stagnation != -1
        ):
            self.should_stop = True

        # Select parents
        parents = self._select_parents(fitness_scores)

        # Create offspring through crossover
        offspring = self._crossover(parents)

        # Mutate offspring
        offspring = self._mutate(offspring)

        elites = np.array(
            [
                self.population[i]
                for i in np.argsort(-fitness_scores)[: self.config.num_elites]
            ]
        )

        # Create new population
        self.population = np.vstack([elites, offspring])
        self.generation += 1
        self.stats.update(
            {
                "fitness/best": self.best_fitness,
                "fitness/mean": np.mean(fitness_scores),
                "fitness/std": np.std(fitness_scores),
            }
        )
        return (self.best_fitness, np.mean(fitness_scores), np.std(fitness_scores))

    def log_stats(self, generation: int, extra_stats: dict | None = None):
        action_freq_stats = {
            f"action_freq/{action}": np.mean(self.population == action)
            for action in range(self.config.num_actions)
        }
        best_individual_freq_stats = {
            f"best_individual_freq/{action}": np.mean(self.best_solution == action)
            for action in range(self.config.num_actions)
        }
        wandb.log(
            {
                "generation": generation,
                **self.stats,
                **{f"time/{k}": v for k, v in self.step_times.items()},
                **(extra_stats or {}),
                **action_freq_stats,
                **best_individual_freq_stats,
            }
        )


def run_curriculum_ga(
    env_builder: Callable,
    curriculum: List[int],
    *,
    old_gene_mutation_rate: float = 0.01,
    new_gene_mutation_rate: float = 0.1,
    normal_mut_at_last_stage: bool = False,
    population_size: int = 100,
    save_dir: Path,
    use_parallel: bool = False,
    num_jobs: int = -4,
    generations_per_stage: list[int] | int = 200,
    max_stagnation: int = 50,
    end_rollout_fn: Callable | None = None,
    final_selection_rate: float = 0.1,
    start_selection_rate: float = 1.0,
    use_champions: bool = False,
) -> GeneticAlgorithm:
    """Run curriculum-based genetic algorithm optimization

    Args:
        curriculum: List of sequence lengths for each curriculum stage
        old_gene_mutation_rate: Mutation rate for old genes
        new_gene_mutation_rate: Mutation rate for new genes
    """
    if isinstance(generations_per_stage, int):
        generations_per_stage = [generations_per_stage] * len(curriculum)
    if len(generations_per_stage) == 1:
        generations_per_stage = generations_per_stage * len(curriculum)

    last_population = None
    last_ga = None
    last_seq_length = 0

    for stage, (seq_length, num_generations) in enumerate(
        zip(curriculum, generations_per_stage)
    ):
        print(f"\nStarting curriculum stage {stage + 1}/{len(curriculum)}")
        print(f"Sequence length: {seq_length}")

        mutation_prob = np.full(seq_length, new_gene_mutation_rate)
        if stage < len(curriculum) - 1 or not normal_mut_at_last_stage:
            mutation_prob[:last_seq_length] = old_gene_mutation_rate
        config = GAConfig(
            num_generations=num_generations,
            population_size=population_size,
            mutation_prob=mutation_prob,
            sequence_length=seq_length,
            use_parallel=use_parallel,
            num_jobs=num_jobs,
            max_stagnation=max_stagnation,
            end_rollout_fn=end_rollout_fn,
            final_selection_rate=final_selection_rate,
            initial_selection_rate=start_selection_rate,
        )

        ga = GeneticAlgorithm(config, env_builder)
        if use_champions:
            ga.population[:len(CHAMPIONS)] = CHAMPIONS[:, :seq_length]

        # Initialize population from previous curriculum stage if available
        if last_population is not None:
            new_population = np.zeros((population_size, seq_length))
            prev_length = curriculum[stage - 1]

            # Copy over previous solutions
            new_population[:, :prev_length] = last_population

            # Randomly initialize new timesteps
            new_population[:, prev_length:] = np.random.randint(
                0, config.num_actions, size=(population_size, seq_length - prev_length)
            )

            ga.population = new_population

        # Run optimization for this curriculum stage
        old_best = 0
        for gen in tqdm(range(num_generations), desc=f"Stage {stage+1} Generation"):
            best_fitness, mean_fitness, std_fitness = ga.step()

            # Check for early stopping
            if ga.should_stop:
                print(
                    f"\nEarly stopping at generation {gen} due to {ga.config.max_stagnation} generations without improvement"
                )
                break

            if best_fitness > old_best:
                old_best = best_fitness
                print(f"Generation {gen}: New best fitness: {best_fitness}")

            # Log metrics
            ga.log_stats(
                gen,
                extra_stats={
                    "curriculum_stage": stage,
                    "sequence_length": seq_length,
                },
            )

        # Save intermediate results
        save_path = save_dir / f"ga_curriculum_stage_{stage}_{int(time.time())}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "solution": ga.best_solution.tolist(),
                    "fitness": float(ga.best_fitness),
                    "curriculum_stage": stage,
                },
                f,
            )

        # Update for next stage
        last_population = ga.population
        last_ga = ga

    return last_ga


END_ROLLOUT_FN_BUILDER_MAP = {
    "random": lambda a: (lambda _obs: np.random.randint(0, 4)),
    "constant": lambda a: (lambda _obs: a),
    None: lambda a: None,
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--num-generations", type=int, nargs="+", default=[100, 200, 300, 400]
    )
    parser.add_argument("--population-size", type=int, default=300)
    parser.add_argument("--mutation-prob", type=float, default=0.1)
    parser.add_argument("--domain-randomization", "-r", action="store_true")
    parser.add_argument(
        "--use-parallel", action="store_true", help="Enable parallel processing"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=-4,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="top_n",
        choices=["tournament", "top_n", "roulette"],
        help="Strategy for parent selection",
    )
    parser.add_argument("--name", "-n", type=str, default=generate_slug(2))
    parser.add_argument(
        "--curriculum",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="Comma-separated list of sequence lengths",
    )
    parser.add_argument(
        "--old-gene-mutation-rate",
        "--old-mut",
        type=float,
        default=0.1,
        help="Mutation rate for previously learned genes",
    )
    parser.add_argument(
        "--new-gene-mutation-rate",
        "--new-mut",
        type=float,
        default=0.3,
        help="Mutation rate for new genes",
    )
    parser.add_argument(
        "--max-stagnation",
        type=int,
        default=50,
        help="Maximum generations without improvement before early stopping",
    )
    parser.add_argument(
        "--normal-mut-at-last-stage",
        action="store_true",
        help="Normal mutation at last stage",
    )
    parser.add_argument(
        "--end-rollout-fn",
        type=str,
        default=None,
        choices=["random", "constant"],
        help="Function to use for end rollout",
    )
    parser.add_argument(
        "--end-rollout-constant",
        type=int,
        default=0,
        help="Action to use for end rollout if end_rollout_fn is constant",
    )
    parser.add_argument(
        "--start-selection-rate",
        type=float,
        default=1.0,
        help="Start selection rate for the first stage",
    )
    parser.add_argument(
        "--final-selection-rate",
        type=float,
        default=0.1,
        help="Final selection rate for the last stage",
    )
    parser.add_argument(
        "--use-champions",
        action="store_true",
        help="Use champion solution for initial population",
    )
    args = parser.parse_args()
    assert not args.domain_randomization, "Domain randomization is not supported"

    # Initialize wandb
    run_name = f"ga-{'rnd' if args.domain_randomization else 'det'}-{int(time.time())}-{args.name}"
    wandb.init(project="ga-hiv", name=run_name)

    # Setup environment
    env_kwargs = dict(
        domain_randomization=args.domain_randomization,
        normalize_reward=False,
        num_frames=1,
        normalize_observation=True,
        time_aware_observation=False,
        last_action_wrapper=False,
    )

    env_b = partial(env_builder, **env_kwargs, use_slow_env=False)

    # Parse curriculum sequence lengths
    ga_instance = run_curriculum_ga(
        env_b,
        curriculum=args.curriculum,
        population_size=args.population_size,
        old_gene_mutation_rate=args.old_gene_mutation_rate,
        new_gene_mutation_rate=args.new_gene_mutation_rate,
        save_dir=Path("models/ga") / run_name,
        use_parallel=args.use_parallel,
        num_jobs=args.num_processes,
        generations_per_stage=args.num_generations,
        max_stagnation=args.max_stagnation,
        end_rollout_fn=END_ROLLOUT_FN_BUILDER_MAP[args.end_rollout_fn](
            args.end_rollout_constant
        ),
        normal_mut_at_last_stage=args.normal_mut_at_last_stage,
        final_selection_rate=args.final_selection_rate,
        start_selection_rate=args.start_selection_rate,
        use_champions=args.use_champions,
    )

    # Evaluate best solution
    print(f"Best solution fitness: {ga_instance.best_fitness}")

    with open(Path("models/ga") / run_name / "ga_solution.json", "w") as f:
        json.dump(
            {
                "best_solution": list(ga_instance.best_solution),
                "best_fitness": float(ga_instance.best_fitness),
            },
            f,
        )

    # eval on fast and slow env
    env = env_builder(**env_kwargs, use_slow_env=False)
    env_slow = env_builder(**env_kwargs, use_slow_env=True)
    env.reset()
    env_slow.reset()
    return_env = 0
    return_env_slow = 0
    for action in ga_instance.best_solution:
        obs, reward, terminated, truncated, info = env.step(int(action))
        obs_slow, reward_slow, terminated_slow, truncated_slow, info_slow = (
            env_slow.step(int(action))
        )
        return_env += reward
        return_env_slow += reward_slow
    print(f"Return env: {return_env:.2e}, Return env slow: {return_env_slow:.2e}")

    wandb.finish()
