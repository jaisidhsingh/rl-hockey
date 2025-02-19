import numpy as np
import torch
import hockey.hockey_env as h_env
import gymnasium as gym

from sac.sac import *
from td3.td3 import *


def load_agent(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = torch.load(path, map_location=device)
    return agent

def load_basic_opponent():
    agent = h_env.BasicOpponent(weak=False)
    return agent


def play(player1, player2, num_episodes=1000, render=False):
    env = h_env.HockeyEnv()

    rewards = []

    for i in range(num_episodes):
        one_starting = np.random.choice([True, False])
        obs, info = env.reset(one_starting=one_starting)
        obs_agent2 = env.obs_agent_two()
        for j in range(250):
            if render:
                env.render(mode="human")
            a1 = player1.select_action(obs)
            a2 = player2.select_action(obs_agent2)
            obs, r, d, t, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            if d or t:
                rewards.append(r)
                break
    
    print("Average reward", np.mean(rewards))
    env.close()


def test(player, num_episodes=1000, render=False):
    env = h_env.HockeyEnv_BasicOpponent()

    rewards = []
    for i in range(num_episodes):
        one_starting = np.random.choice([True, False])
        obs, info = env.reset(one_starting=one_starting)
        for j in range(250):
            if render:
                env.render(mode="human")
            action = player.select_action(obs)
            obs, r, d, t, info = env.step(action)
            if d or t:
                rewards.append(r)
                break
    
    print("Average reward", np.mean(rewards))
    env.close()


def main():
    agent1_path = "agents/sac_agent_600000.pt"
    agent2_path = "agents/td3_agent.pt"
    player1 = load_agent(agent1_path)
    player2 = load_agent(agent2_path)

    player1.device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    
    player2.device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")

    play(player1, player2)

    # test(player1)


if __name__ == "__main__":
    main()