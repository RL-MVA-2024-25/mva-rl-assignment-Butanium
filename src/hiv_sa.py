import numpy as np
from tqdm.rich import tqdm
import json
import math
from env_hiv_fast import FastHIVPatient
from hiv_mcts import MCTS_CHAMPION


class SimulatedAnnealing:
    def __init__(
        self, env, max_steps=200, initial_temp=100.0, min_temp=0.1, cooling_rate=0.95
    ):
        self.env = env
        self.max_steps = max_steps
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate

    def evaluate_sequence(self, action_sequence):
        """Evaluate a sequence of actions and return total reward."""
        self.env.reset()
        total_reward = 0

        for action in action_sequence:
            _, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            if done:
                break

        return total_reward

    def generate_neighbor(self, current_sequence):
        """Generate a neighboring solution with smarter modifications."""
        neighbor = current_sequence.copy()
        
        # Randomly choose modification strategy
        strategy = np.random.choice(['single', 'consecutive', 'small_change'])
        
        if strategy == 'single':
            # Original strategy
            idx = np.random.randint(len(neighbor))
            neighbor[idx] = np.random.randint(4)
        
        elif strategy == 'consecutive':
            # Modify 2-3 consecutive actions
            start_idx = np.random.randint(len(neighbor) - 2)
            length = np.random.randint(2, 4)
            for i in range(start_idx, min(start_idx + length, len(neighbor))):
                neighbor[i] = np.random.randint(4)
        
        else:  # small_change
            # Small perturbation of existing action
            idx = np.random.randint(len(neighbor))
            current_action = neighbor[idx]
            # Choose from adjacent actions (with wrapping)
            neighbor[idx] = (current_action + np.random.choice([-1, 1])) % 4
        
        return neighbor

    def acceptance_probability(self, old_reward, new_reward, temperature):
        """Calculate probability of accepting worse solution."""
        if new_reward > old_reward:
            return 1.0
        return math.exp((new_reward - old_reward) / temperature)

    def optimize(self, n_iterations=1000, initial_sequence=None, n_restarts=5):
        """Optimize with multiple restarts."""
        best_overall_sequence = None
        best_overall_reward = float('-inf')
        
        for restart in range(n_restarts):
            temperature = self.initial_temp
            if initial_sequence is None:
                current_sequence = np.random.randint(0, 4, size=self.max_steps)
            else:
                # Slightly perturb the initial sequence for each restart
                current_sequence = initial_sequence.copy()
                mask = np.random.random(size=len(current_sequence)) < 0.1  # 10% mutation
                current_sequence[mask] = np.random.randint(0, 4, size=mask.sum())
                
            current_reward = self.evaluate_sequence(current_sequence)

            best_sequence = current_sequence.copy()
            best_reward = current_reward

            pbar = tqdm(range(n_iterations), desc="Optimizing")
            for _ in pbar:
                if temperature < self.min_temp:
                    break

                # Generate and evaluate neighbor
                neighbor_sequence = self.generate_neighbor(current_sequence)
                neighbor_reward = self.evaluate_sequence(neighbor_sequence)

                # Decide if we should accept the neighbor
                if (
                    self.acceptance_probability(
                        current_reward, neighbor_reward, temperature
                    )
                    > np.random.random()
                ):
                    current_sequence = neighbor_sequence
                    current_reward = neighbor_reward

                    # Update best if we found a better solution
                    if current_reward > best_reward:
                        best_reward = current_reward
                        best_sequence = current_sequence.copy()
                        pbar.set_postfix({"best_reward": f"{best_reward:.2f}"})

                # Cool down
                temperature *= self.cooling_rate

            if best_reward > best_overall_reward:
                best_overall_reward = best_reward
                best_overall_sequence = best_sequence.copy()
                print(f"New best reward: {best_overall_reward:.2e}")
        
        return best_overall_sequence.tolist(), best_overall_reward


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    # Create environment
    env = FastHIVPatient(domain_randomization=args.domain_randomization)

    # Create and run simulated annealing
    sa = SimulatedAnnealing(
        env,
        max_steps=200,
        initial_temp=50.0,
        min_temp=0.01,
        cooling_rate=0.98
    )

    best_actions, best_reward = sa.optimize(
        n_iterations=args.iterations,
        initial_sequence=np.array(MCTS_CHAMPION),
        n_restarts=5
    )

    print("\nBest performing sequence:")
    print(f"Reward: {best_reward:.2f}")
    print("Actions:", best_actions)

    # Verify best sequence
    obs, _ = env.reset()
    total_reward = 0
    for action in best_actions:
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Verification reward: {total_reward:.2e}")

    # Save results
    with open("best_sol.json", "w") as f:
        json.dump({"best_actions": best_actions, "best_reward": best_reward}, f)


if __name__ == "__main__":
    main()
