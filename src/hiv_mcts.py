import argparse
from pathlib import Path
import numpy as np
import math
from threading import Lock
import time

# from tqdm.auto import tqdm
from tqdm.rich import tqdm
from env_hiv_fast import FastHIVPatient
from joblib import Parallel, delayed
import json
from env_hiv import SlowHIVPatient
import wandb
from wandb import plot
from coolname import generate_slug

MCTS_CHAMPION = [
    3,
    3,
    0,
    2,
    1,
    3,
    1,
    3,
    2,
    2,
    2,
    2,
    2,
    3,
    2,
    3,
    3,
    3,
    3,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    0,
    1,
    0,
    0,
    0,
    1,
    2,
    1,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    0,
    3,
    1,
    1,
    2,
    0,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    0,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    0,
    1,
    2,
    0,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    2,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
]


class MCTSNode:
    def __init__(self, state, reward=0, parent=None, action=None, max_depth=200):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.lock = Lock()  # Add lock for thread-safe updates
        self.depth = 0 if parent is None else parent.depth + 1
        self.cumulative_reward = (
            reward + self.parent.cumulative_reward if parent else reward
        )
        self.untried_actions = list(range(4)) if self.depth < max_depth else []
        self.is_terminal = self.depth == max_depth
        self.best_return = 0
        self.max_depth = max_depth

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def node_value(self):
        if self.is_terminal:
            return self.best_return
        else:
            return self.value / self.visits

    def check_terminal(self):
        self.is_terminal = self.is_terminal or (
            len(self.untried_actions) == 0
            and np.all([child.is_terminal for child in self.children.values()])
        )
        return self.is_terminal

    def best_child(self, c_param=1.414):
        choices = [
            (
                ((np.log10(max(1, child.node_value()))) / 10)
                + c_param * math.sqrt(2 * math.log(self.visits) / child.visits),
                action,
                child,
            )
            for action, child in self.children.items()
        ]
        _, action, child = max(choices)
        return action, child

    def expand(self, action, next_state, reward):
        child = MCTSNode(
            next_state, reward, parent=self, action=action, max_depth=self.max_depth
        )
        self.untried_actions.remove(action)
        self.children[action] = child
        return child

    def update(self, reward):
        with self.lock:
            self.visits += 1
            self.value += reward
            self.best_return = max(self.best_return, reward)


def single_rollout(sim_env_args, max_depth):
    # Simulation (rollout)
    sim_env = FastHIVPatient.from_state(*sim_env_args)
    cumulative_reward = 0
    depth = 0

    while depth < max_depth:
        action = np.random.randint(4)  # Random policy for rollout
        obs, reward, done, _, _ = sim_env.step(action)
        cumulative_reward += reward
        depth += 1
        if done:
            break
    return cumulative_reward


class MCTS:
    def __init__(
        self,
        env,
        num_simulations=1000,
        max_depth=200,
        use_parallel=True,
        n_jobs=-4,
        n_rollouts=20,
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        self.n_rollouts = n_rollouts

    def _rollout(self, max_depth):
        if self.use_parallel:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(single_rollout)(self.env.clone_args(), max_depth)
                for _ in range(self.n_rollouts)
            )
            return np.array(results)
        else:
            return np.array(
                [
                    single_rollout(self.env.clone_args(), max_depth)
                    for _ in range(self.n_rollouts)
                ]
            )

    def search(self, initial_state):
        if isinstance(initial_state, MCTSNode):
            root = initial_state
        else:
            root = MCTSNode(initial_state, max_depth=self.max_depth)
        total_timesteps = 0
        max_depth_explored = root.depth
        for _ in range(self.num_simulations):
            node = root
            if node.check_terminal():
                break
            # Selection
            while (
                node.is_fully_expanded() and node.children and not node.check_terminal()
            ):
                action, node = node.best_child()
            self.env.state_vec = node.state
            # Expansion
            if not node.is_fully_expanded():
                action = np.random.choice(node.untried_actions)
                _, reward, _, _, _ = self.env.step(action)
                total_timesteps += 1
                node = node.expand(action, self.env.state_vec, reward)
            max_depth_explored = max(max_depth_explored, node.depth)
            if node.is_terminal:
                returns = [node.best_return]
            else:
                returns = (
                    self._rollout(self.max_depth - node.depth) + node.cumulative_reward
                )
                total_timesteps += (self.max_depth - node.depth) * self.n_rollouts
            # Backpropagation
            while node:
                node.visits += len(returns)
                node.value += np.sum(returns)
                node.best_return = max(node.best_return, np.max(returns))
                node = node.parent
        print(f"total timesteps: {total_timesteps}")
        print(f"max depth explored: {max_depth_explored}")
        wandb.log(
            {
                "total_timesteps": total_timesteps,
                "max_depth_explored": max_depth_explored,
            },
            commit=False,
        )
        # Return best action from root
        return root.best_child(c_param=0)  # c_param=0 for exploitation only


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--num-simulations-fst", type=int, default=1000)
    parser.add_argument("--num-simulations-lst", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=200)
    parser.add_argument("--no-parallel", action="store_false", dest="use_parallel")
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name", type=str, default=generate_slug(2))
    args = parser.parse_args()
    name = str(int(time.time())) + "_" + args.name
    # Create multiple environments for parallel MCTS
    env = FastHIVPatient(domain_randomization=args.domain_randomization)
    if not args.debug:
        wandb.init(project="hiv-mcts", name=name)
    # Create MCTS instance
    mcts = MCTS(
        env,
        num_simulations=args.num_simulations_fst,
        max_depth=args.max_depth,
        use_parallel=args.use_parallel,
        n_rollouts=args.n_rollouts,
        n_jobs=args.n_jobs,
    )
    save_path = Path(f"models/mcts/{name}")
    save_path.mkdir(parents=True, exist_ok=True)
    # Run multiple episodes to find best performing sequence
    n_episodes = args.num_episodes
    best_reward = float("-inf")
    best_actions = []
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_actions = []
            total_reward = 0
            root = MCTSNode(obs, max_depth=args.max_depth)
            # Run episode
            for step in tqdm(
                range(args.episode_length), desc="Steps"
            ):  # Max steps per episode
                action, root = mcts.search(root)
                parent = root.parent
                root.parent = None
                stats = {
                    "reward": np.log10(max(1, total_reward)),
                    "step": step,
                    "mean_reward/chosen_node": np.log10(
                        max(1, root.value / root.visits)
                    ),
                    "mean_reward/mean_node": np.log10(
                        np.mean(
                            [
                                child.value / child.visits
                                for child in parent.children.values()
                            ]
                        )
                    ),
                    "mean_reward/worst_node": np.log10(
                        np.min(
                            [
                                child.value / child.visits
                                for child in parent.children.values()
                            ]
                        )
                    ),
                    "best_return/chosen_node": np.log10(max(1, root.best_return)),
                    "best_return/mean_node": np.log10(
                        np.mean(
                            [child.best_return for child in parent.children.values()]
                        )
                    ),
                    "best_return/worst_node": np.log10(
                        np.min(
                            [child.best_return for child in parent.children.values()]
                        )
                    ),
                    "visits/chosen_node": root.visits,
                    "ratio/chosen_node": root.visits / parent.visits,
                    "ratio/mean_node": np.mean(
                        [
                            child.visits / parent.visits
                            for child in parent.children.values()
                        ]
                    ),
                    "ratio/worst_node": np.min(
                        [
                            child.visits / parent.visits
                            for child in parent.children.values()
                        ]
                    ),
                }
                if args.debug:
                    print(stats)
                else:
                    wandb.log(stats)
                episode_actions.append(int(action))
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                mcts.num_simulations = args.num_simulations_lst

            print(f"Episode {episode + 1}/{n_episodes}: Reward = {total_reward:.2f}")

            # Update best sequence if current episode performed better
            if total_reward > best_reward:
                best_reward = total_reward
                best_actions = episode_actions.copy()
    except Exception as e:
        wandb.alert(title=f"Error at step {step}", text=str(e))
        print(f"best_actions: {best_actions}\n episode_actions: {episode_actions}")
        with open(save_path / "best_sol.json", "w") as f:
            json.dump({"best_actions": episode_actions, "best_reward": total_reward}, f)
        raise e

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
    print(f"Verification reward: {total_reward:.2f}")
    env_slow = SlowHIVPatient()
    obs, _ = env_slow.reset()
    for action in best_actions:
        obs, reward, done, _, _ = env_slow.step(action)
        total_reward += reward
        if done:
            break
    print(f"Verification reward on slow env: {total_reward:.2f}")
    with open(save_path / "best_sol.json", "w") as f:
        json.dump({"best_actions": best_actions, "best_reward": best_reward}, f)
