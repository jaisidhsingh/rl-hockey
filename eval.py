import numpy as np
import torch
import hockey.hockey_env as h_env
import gymnasium as gym
from tqdm import tqdm

from sac.sac import *
from td3.td3 import *

from utils.utils import load_agent, play
from utils.agent_picker import AgentPicker


def play_hockey(player1, player2, num_episodes=1000, render=False):
    rewards = []
    results = []
    for i in tqdm(range(num_episodes), desc="Playing hockey:"):
        reward, result = play(player1, player2, render=render)
        rewards.append(reward)
        results.append(result)
    print(f"Player 1 win ratio: {np.mean(results)}")
    print(f"Player 1 avg reward: {np.mean(rewards)}")


def test_all_agents(player, num_episodes=1000, render=False):
    agent_picker = AgentPicker("agents/checkpoints")

    agent_picker.init_agents(player)

    for i in tqdm(range(num_episodes), desc="Playing against other agents:"):
        opponent, idx = agent_picker.pick_agent()
        reward, result = play(player, opponent, render=render)
        agent_picker.update_agent_info(idx, result)
    
    agent_picker.show_agents()


def main():
    agent1_path = "agents/sac_agent_sp150000.pt"
    agent2_path = "agents/sac_agent.pt"

    player1 = load_agent(agent1_path)
    player2 = load_agent(agent2_path)

    # play_hockey(player1, player2, num_episodes=100, render=True)
    test_all_agents(player1, num_episodes=1000)


if __name__ == "__main__":
    main()