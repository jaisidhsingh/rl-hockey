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


def test_all_agents(player, opponent_dir, num_episodes=1000, render=False, uniform=False):
    agent_picker = AgentPicker(opponent_dir)

    agent_picker.init_agents(player)

    for i in tqdm(range(num_episodes), desc="Playing against other agents:"):
        opponent, idx = agent_picker.pick_agent(uniform=uniform)
        reward, result = play(player, opponent, render=render)
        agent_picker.update_agent_info(idx, result)
    
    agent_picker.show_agents()


def main():
    agent1_path = "agents/td3_agent_sp1M(2)_1.pt"
    agent2_path = "agents/sac_agent_strongest.pt"

    player1 = load_agent(agent1_path)
    player2 = load_agent(agent2_path)

    play_hockey(player1, player2, num_episodes=1000)
    # test_all_agents(player1, opponent_dir="agents", num_episodes=1000, uniform=True)


if __name__ == "__main__":
    main()