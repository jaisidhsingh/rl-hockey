import os
import numpy as np
import torch
from tqdm import tqdm

from sac import *
from td3 import *


def load_agent(path):
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if torch.backends.mps.is_available() 
                         else "cpu")
    agent = torch.load(path, map_location=device, weights_only=False)
    agent.device = device
    return agent


class AgentPicker:
    """
    Picks an agent from a directory of agents. The probability of picking each agent depends on the performance of the player against this agent.
    If the player wins frequently against an agent, the agent will be picked less often.
    """
    def __init__(self, agent_dir):
        self.agent_names = [f for f in os.listdir(agent_dir) if f.endswith('.pt')]
        self.agents = [load_agent(os.path.join(agent_dir, f)) for f in self.agent_names]
        self.agent_play_counts = np.zeros(len(self.agents))
        self.agent_weights = np.zeros(len(self.agents))

        print("Loaded the following agents: ", self.agent_names)
    
    def get_win_ratio(self):
        return self.agent_weights / self.agent_play_counts
    
    def add_agent(self, agent, agent_name):
        self.agents.append(agent)
        self.agent_names.append(agent_name)
        self.agent_weights.append(0)
    
    def get_all_agents(self):
        return self.agents

    def pick_agent(self, uniform=False, scale=0.03):
        if uniform:
            idx = np.random.choice(range(len(self.agents)))
        else:
            logits = self.agent_weights - np.max(self.agent_weights)
            exp_logits = np.exp(logits * scale)
            probs = exp_logits / np.sum(exp_logits)
            idx = np.random.choice(range(len(self.agents)), p=probs)
        return self.agents[idx], idx

    def update_agent_info(self, idx, result):
        """
        Update the weights of the agent based on the result of the game. (1 for win, -1 for loss, 0 for draw)
        Note that we use the negative of the result, because the result that is passed in belongs to the opposite agent.
        Maintains a running average of the negative results.
        """
        self.agent_play_counts[idx] += 1
        self.agent_weights[idx] -= result

    def show_agents(self):
        win_ratios = self.get_win_ratio()
        sorted_idx = np.argsort(win_ratios)[::-1]
        for i in sorted_idx:
            print(f"Agent {i}: {self.agent_names[i]}, Avg result: {win_ratios[i]:.2f}, Play count: {self.agent_play_counts[i]}")



