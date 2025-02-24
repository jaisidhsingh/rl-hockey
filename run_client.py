from __future__ import annotations

import argparse
import uuid
import torch
import hockey.hockey_env as h_env
import numpy as np


from sac.sac import *
from td3.td3 import *
from utils.utils import load_agent
from comprl.client import Agent, launch_client


class TD3Agent(Agent):
    """A hockey agent trained using TD3 + self-play."""

    def __init__(self, agent_checkpoint: str) -> None:
        super().__init__()
        self.hockey_agent = load_agent(agent_checkpoint)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.select_action(np.array(observation)).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class SACAgent(Agent):
    """A hockey agent trained using SAC + self-play."""

    def __init__(self, agent_checkpoint: str) -> None:
        super().__init__()
        self.hockey_agent = load_agent(agent_checkpoint)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.select_action(np.array(observation), evaluate=True).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["sac_sp", "td3_sp"],
        default="sac_sp",
        help="Which agent to use.",
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "td3_sp":
        agent = TD3Agent(args.checkpoint_path)
        print(f"Initialized TD3 agent from {args.checkpoint_path}")
    elif args.agent == "sac_sp":
        agent = SACAgent(args.checkpoint_path)
        print(f"Initialized SAC agent from {args.checkpoint_path}")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
