import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from src.env_hiv_fast import FastHIVPatient
from tqdm import tqdm
import seaborn as sns
import json
import pickle


def heuristic_pallier():
    env = FastHIVPatient()
    env.reset()

    def continue_treatment(state_vec, current_time, current_reward):
        if current_time == 200:
            return current_time, current_reward
        env = FastHIVPatient()
        env.state_vec = state_vec
        for step in range(current_time, 200):
            _obs, reward, _, _, _ = env.step(0)
            current_reward += reward
        return current_time, current_reward

    # Collect states and rewards as we go
    states_and_times = []
    cum_reward = 0

    for step in range(200):
        states_and_times.append((env.state_vec.copy(), step, cum_reward))
        _obs, reward, _, _, _ = env.step(3)
        cum_reward += reward

    # Run parallel simulations
    results = Parallel(n_jobs=-1)(
        delayed(continue_treatment)(state, time, reward)
        for state, time, reward in states_and_times
    )

    # Convert results to dictionary
    results_dict = {t: r for t, r in results}
    results_dict[200] = cum_reward  # Add the final result

    # plot results
    plt.plot(results_dict.keys(), results_dict.values())
    plt.savefig("results_heuristic.png")


def greedy_heuristic(consecutive_actions=1, num_watch_steps=5, domain_randomization=False):
    env = FastHIVPatient(domain_randomization=domain_randomization)
    env.reset()
    cum_reward = 0
    actions = []
    for step in range(200):
        rewards = []
        for action in range(4):
            env_copy = env.clone()
            action_reward = 0
            for _ in range(consecutive_actions):
                _, reward, _, _, _ = env_copy.step(action)
                action_reward += reward
            for _ in range(num_watch_steps):
                _, reward, _, _, _ = env_copy.step(0)
                action_reward += reward
            rewards.append(action_reward)
        best_action = int(np.argmax(rewards))
        _, reward, _, _, _ = env.step(best_action)
        actions.append(best_action)
        cum_reward += reward
    print(
        f"Delayed of {consecutive_actions} actions:\ncumulative reward: {cum_reward:.2e}"
    )
    return consecutive_actions, cum_reward, actions


def main_greedy_cons(num_watch_steps=10):
    results = Parallel(n_jobs=-1)(
        delayed(greedy_heuristic)(consecutive_actions, num_watch_steps)
        for consecutive_actions in range(1, 20)
    )
    res_dict = {}
    best_actions = []
    best_cum_reward = 0
    best_consecutive_actions = 0
    for consecutive_actions, cum_reward, actions in results:
        res_dict[consecutive_actions] = (cum_reward, actions)
        if cum_reward > best_cum_reward:
            best_cum_reward = cum_reward
            best_actions = actions
            best_consecutive_actions = consecutive_actions
    print("\n\n===============")
    print(
        f"Best cumulative reward: {best_cum_reward:.2e}\nBest actions: {best_actions}\nBest consecutive actions: {best_consecutive_actions}"
    )
    print("===============")
    plt.plot(res_dict.keys(), [res[0] for res in res_dict.values()])
    plt.savefig(f"plots/results_greedy_{num_watch_steps}.png")
    return best_consecutive_actions, best_cum_reward, best_actions, res_dict


def main_greedy():
    global best_sol_nws, best_sol_consecutive_actions, best_sol_cum_reward, best_sol_actions, res_dicts
    results = []
    num_watch_steps_list = [0, 5, 10, 20, 30, 40, 50]
    for num_watch_steps in num_watch_steps_list:
        results.append(main_greedy_cons(num_watch_steps))

    best_sol_nws = 0
    best_sol_cum_reward = 0
    best_sol_consecutive_actions = 0
    best_sol_actions = []
    for nws, (best_consecutive_actions, best_cum_reward, best_actions, res_dict) in zip(
        num_watch_steps_list, results
    ):
        if best_cum_reward > best_sol_cum_reward:
            best_sol_cum_reward = best_cum_reward
            best_sol_consecutive_actions = best_consecutive_actions
            best_sol_actions = best_actions
            best_sol_nws = nws
    print("\n\n**************")
    print(
        f"Best solution: {best_sol_nws} {best_sol_consecutive_actions} {best_sol_cum_reward:.2e} {best_sol_actions}"
    )
    print("**************")
    try:
        with open("results_greedy.json", "w") as f:
            json.dump(results, f)
    except Exception as e:
        print(f"Error saving results: {e}")
        try:
            with open("results_greedy.pkl", "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving results with pickle: {e}")
    # plot as a heatmap
    sns.heatmap(
        {
            nws: {
                consecutive_actions: cum_reward
                for consecutive_actions, (cum_reward, _) in res_dict.items()
            }
            for nws, res_dict in zip(num_watch_steps_list, results)
        }
    )
    plt.savefig("plots/results_greedy_heatmap.png")
    return (
        best_sol_nws,
        best_sol_consecutive_actions,
        best_sol_cum_reward,
        best_sol_actions,
        results,
    )


def main_greedy_domain_randomization(consecutive_actions=1, num_watch_steps=5):
    results = Parallel(n_jobs=-1)(
        delayed(greedy_heuristic)(consecutive_actions, num_watch_steps, True)
        for _ in range(100)
    )
    print(f'mean reward: {np.mean([res[1] for res in results])}')
    with open("results_greedy_domain_randomization.json", "w") as f:
        json.dump(results, f)
    
    # plot histogram of results
    plt.hist([res[1] for res in results])
    plt.savefig("plots/results_greedy_domain_randomization.png")

if __name__ == "__main__":
    # heuristic_pallier()
    # main_greedy()
    main_greedy_domain_randomization()
