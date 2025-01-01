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
from typing import List, Tuple
from joblib import Parallel, delayed
from coolname import generate_slug
from src.common import env_builder

@dataclass
class GAConfig:
    num_generations: int = 1000
    population_size: int = 50
    num_parents: int = 10
    mutation_prob: float = 0.1
    sequence_length: int = 200
    num_actions: int = 4
    use_parallel: bool = False
    num_jobs: int = 4
    selection_strategy: str = "tournament"
    parallel_mutation: bool = False

class GeneticAlgorithm:
    def __init__(self, config: GAConfig, env_builder: Callable):
        self.config = config
        self.env_builder = env_builder
        self.population = self._initialize_population()
        self.best_fitness = float('-inf')
        self.best_solution = None
        self.generation = 0
        self.last_generation_fitness = []
        self.step_times = {}
        self._selection_strategies = {
            "tournament": self._tournament_selection,
            "top_n": self._top_n_selection,
            "roulette": self._roulette_selection
        }

    def _initialize_population(self) -> np.ndarray:
        return np.random.randint(
            0, self.config.num_actions, 
            size=(self.config.population_size, self.config.sequence_length)
        )

    def _evaluate_solution(self, solution: np.ndarray) -> float:
        env = self.env_builder()
        obs, _ = env.reset()
        total_reward = 0

        for action in solution:
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
            fitness_scores = [self._evaluate_solution(solution) for solution in self.population]
        self.step_times['evaluate'] = time.time() - start_time
        return np.array(fitness_scores)

    @staticmethod
    def _evaluate_solution_wrapper(solution: np.ndarray, env_builder: Callable) -> float:
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

    def _select_parents(self, fitness_scores: np.ndarray) -> np.ndarray:
        start_time = time.time()
        selection_func = self._selection_strategies.get(self.config.selection_strategy)
        if selection_func is None:
            raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")
        
        selected_parents = selection_func(fitness_scores)
        self.step_times['select'] = time.time() - start_time
        return selected_parents

    def _tournament_selection(self, fitness_scores: np.ndarray) -> np.ndarray:
        selected_parents = []
        for _ in range(self.config.num_parents):
            tournament_idx = np.random.choice(len(self.population), size=3, replace=False)
            tournament_fitness = fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected_parents.append(self.population[winner_idx])
        return np.array(selected_parents)

    def _top_n_selection(self, fitness_scores: np.ndarray) -> np.ndarray:
        top_indices = np.argsort(fitness_scores)[-self.config.num_parents:]
        return self.population[top_indices]

    def _roulette_selection(self, fitness_scores: np.ndarray) -> np.ndarray:
        # Shift fitness scores to be positive
        shifted_fitness = fitness_scores - np.min(fitness_scores) + 1e-6
        probabilities = shifted_fitness / shifted_fitness.sum()
        selected_indices = np.random.choice(
            len(self.population), 
            size=self.config.num_parents, 
            p=probabilities,
            replace=False
        )
        return self.population[selected_indices]

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        start_time = time.time()
        offspring = []
        num_offspring = self.config.population_size - len(parents)
        
        while len(offspring) < num_offspring:
            parent1, parent2 = random.sample(list(parents), k=2)
            # Single-point crossover
            crossover_point = random.randint(1, self.config.sequence_length-1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring.append(child)

        self.step_times['crossover'] = time.time() - start_time
        return np.array(offspring)

    def _mutate(self, solutions: np.ndarray) -> np.ndarray:
        start_time = time.time()
        if self.config.use_parallel and self.config.parallel_mutation:
            solutions = np.array(Parallel(n_jobs=self.config.num_jobs)(
                delayed(self._mutate_single)(
                    solution, self.config.mutation_prob, self.config.num_actions
                ) for solution in solutions
            ))
        else:
            for solution in solutions:
                self._mutate_single(solution, self.config.mutation_prob, self.config.num_actions)
        self.step_times['mutate'] = time.time() - start_time
        return solutions

    @staticmethod
    def _mutate_single(solution: np.ndarray, mutation_prob: float, num_actions: int) -> np.ndarray:
        # Static method for parallel mutation
        solution = solution.copy()  # Create a copy to avoid modifying the original
        for idx in range(len(solution)):
            if random.random() < mutation_prob:
                solution[idx] = random.randint(0, num_actions - 1)
        return solution

    def step(self) -> Tuple[float, float, float]:
        # Evaluate current population
        fitness_scores = self._evaluate_population()
        self.last_generation_fitness = fitness_scores

        # Update best solution
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_solution = self.population[best_idx].copy()

        # Select parents
        parents = self._select_parents(fitness_scores)

        # Create offspring through crossover
        offspring = self._crossover(parents)

        # Mutate offspring
        offspring = self._mutate(offspring)

        # Create new population
        self.population = np.vstack([parents, offspring])
        self.generation += 1

        return (
            self.best_fitness,
            np.mean(fitness_scores),
            np.std(fitness_scores)
        )

def run_ga(
    env_builder: Callable,
    *,
    num_generations: int = 1000,
    population_size: int = 50,
    num_parents: int = 10,
    mutation_prob: float = 0.1,
    save_dir: Path,
    use_parallel: bool = False,
    num_jobs: int = -4,
) -> GeneticAlgorithm:
    """Run genetic algorithm optimization"""

    config = GAConfig(
        num_generations=num_generations,
        population_size=population_size,
        num_parents=num_parents,
        mutation_prob=mutation_prob,
        use_parallel=use_parallel,
        num_jobs=num_jobs,
    )

    ga = GeneticAlgorithm(config, env_builder)
    
    # Run optimization
    pbar = tqdm(total=num_generations, desc="Generation")
    for gen in range(num_generations):
        best_fitness, mean_fitness, std_fitness = ga.step()
        
        # Log metrics
        wandb.log({
            "fitness/best": best_fitness,
            "fitness/mean": mean_fitness,
            "fitness/std": std_fitness,
            "generation": gen,
            **{f"time/{k}": v for k, v in ga.step_times.items()},
        })
        
        print(f"Generation = {gen}")
        print(f"Fitness = {best_fitness}")
        pbar.update(1)

    # Save best solution
    save_path = save_dir / f"ga_solution_{int(time.time())}.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump({
            "solution": ga.best_solution,
            "fitness": ga.best_fitness,
            "ga_instance": ga,
        }, f)

    return ga


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num-generations", type=int, default=1000)
    parser.add_argument("--sol-per-pop", type=int, default=50)
    parser.add_argument("--num-parents-mating", type=int, default=10)
    parser.add_argument("--mutation-prob", type=float, default=0.1)
    parser.add_argument("--domain-randomization", "-r", action="store_true")
    parser.add_argument("--use-parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--num-processes", type=int, default=20, help="Number of processes for parallel processing")
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="top_n",
        choices=["tournament", "top_n", "roulette"],
        help="Strategy for parent selection"
    )
    parser.add_argument("--name", "-n", type=str, default=generate_slug(2))
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

    # Run GA
    ga_instance = run_ga(
        env_b,
        num_generations=args.num_generations,
        population_size=args.sol_per_pop,
        num_parents=args.num_parents_mating,
        mutation_prob=args.mutation_prob,
        save_dir=Path("models/ga"),
        use_parallel=args.use_parallel,
        num_jobs=args.num_processes,
    )

    # Evaluate best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    print(f"Best solution fitness: {solution_fitness}")

    wandb.finish()
