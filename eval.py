import numpy as np
import torch
import hockey.hockey_env as h_env
import gymnasium as gym
from tqdm import tqdm

from sac.sac import *
from td3.td3 import *
from utils.agent_picker import AgentPicker


def load_agent(path):
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    agent = torch.load(path, map_location=device, weights_only=False)
    agent.device = device
    return agent


def load_basic_opponent():
    agent = h_env.BasicOpponent(weak=False)
    return agent


def test_basic_opponent(player, num_episodes=1000, render=False):
    env = h_env.HockeyEnv_BasicOpponent()

    rewards = []
    for i in tqdm(range(num_episodes), desc="Testing against basic opponent:"):
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
    
    print("Average reward:", np.mean(rewards))
    env.close()


def play(player, opponent, render=False):
    env = h_env.HockeyEnv()
    rewards = 0
    one_starting = np.random.choice([True, False])
    obs, info = env.reset(one_starting=one_starting)
    obs_agent2 = env.obs_agent_two()
    for step in range(250):
        if render:
            env.render(mode="human")
        a1 = player.select_action(obs)
        a2 = opponent.select_action(obs_agent2)
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        obs_agent2 = env.obs_agent_two()
        rewards += r
        if d or t:
            break
    return rewards, info["winner"]


def test_all_agents(player, init_episodes=10, num_episodes=1000, render=False):
    agent_picker = AgentPicker("agents/checkpoints")

    agents = agent_picker.get_all_agents()
    for i, agent in tqdm(enumerate(agents), desc="Initial evaluation of agents:", total=len(agents)):
        for j in range(init_episodes):
            reward, result = play(player, agent, render=render)
            agent_picker.update_agent_info(i, result)

    for i in tqdm(range(num_episodes), desc="Playing against other agents:"):
        opponent, idx = agent_picker.pick_agent()
        reward, result = play(player, opponent, render=render)
        agent_picker.update_agent_info(idx, result)
    
    agent_picker.show_agents()


def main():
    # agent1_path = "agents/td3_agent.pt"
    agent1_path = "agents/checkpoints/td3_agent_1500000.pt"

    player1 = load_agent(agent1_path)
    # player2 = load_agent(agent2_path)

    # play(player1, player2, num_episodes=100)
    test_all_agents(player1, num_episodes=1000)


if __name__ == "__main__":
    main()