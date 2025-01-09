import random
import os
from pathlib import Path
import numpy as np
import torch
from argparse import ArgumentParser
from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    file = Path(f"score_{args.model_name}.txt")
    if not file.is_file():
        seed_everything(seed=42)
        # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
        agent = ProjectAgent(
            # use_lstm=args.use_lstm,
            # model_name=args.model_name,
            # deterministic=args.deterministic,
        )
        agent.load()
        # Evaluate agent and write score.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
        print(f"Score single agent: {score_agent:.3e}")
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
        print(f"Score population agent: {score_agent_dr:.3e}")
        with open(file="score.txt", mode="w") as f:
            f.write(f"{score_agent}\n{score_agent_dr}")
        with open(file=file, mode="w") as f:
            f.write(f"{score_agent:.3e}\n{score_agent_dr:.3e}")
        # launch pytest on grading.py using pytest import
        import pytest

        pytest.main(["grading.py"])

