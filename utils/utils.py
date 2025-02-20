import numpy as np
import torch
import hockey.hockey_env as h_env


def load_agent(path):
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    agent = torch.load(path, map_location=device, weights_only=False)
    agent.device = device
    return agent


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