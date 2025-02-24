import os
import numpy as np

from utils.utils import load_agent, play


class AgentPicker:
    """
    Picks an agent from a directory of agents. The probability of picking each agent depends on the performance of the player against this agent.
    If the player wins frequently against an agent, the agent will be picked less often.
    """
    def __init__(self, agent_dir):
        # Convert to numpy array right from the start
        self.agent_names = np.array([f for f in os.listdir(agent_dir) if f.endswith('.pt')])
        self.agents = np.array([load_agent(os.path.join(agent_dir, f)) for f in self.agent_names])
        self.agent_num_plays = np.zeros(len(self.agents))
        self.agent_results = np.zeros(len(self.agents))
        self.init_episodes = 10
        self.scale = 3

        if len(self.agents) == 0:
            print(f"Warning. No agents were initialized. The directory {agent_dir} might contain no agents.")
        else:
            print("Loaded the following agents: ", self.agent_names)
    
    def init_agents(self, player):
        for idx, agent in enumerate(self.agents):
            for i in range(self.init_episodes):
                reward, result = play(player, agent)
                self.update_agent_info(idx, result)
    
    def add_agent(self, agent, agent_name, player):
        self.agents = np.append(self.agents, [agent])
        self.agent_names = np.append(self.agent_names, [agent_name])
        self.agent_num_plays = np.append(self.agent_num_plays, 0)
        self.agent_results = np.append(self.agent_results, 0)

        for i in range(self.init_episodes):
            reward, result = play(player, agent)
            self.update_agent_info(len(self.agents) - 1, result)
    
    def drop_weakest_agent(self):
        weakest_idx = np.argmin(self.agent_results)
        name = self.agent_names[weakest_idx]
        result = self.agent_results[weakest_idx]
        self.agents = np.delete(self.agents, weakest_idx)
        self.agent_names = np.delete(self.agent_names, weakest_idx)
        self.agent_results = np.delete(self.agent_results, weakest_idx)
        self.agent_num_plays = np.delete(self.agent_num_plays, weakest_idx)
        return name

    def pick_agent(self, uniform=False):
        if uniform:
            idx = np.random.choice(range(len(self.agents)))
        else:
            performances = self.get_performance()
            logits = performances - np.max(performances)
            exp_logits = np.exp(logits * self.scale)
            probs = exp_logits / np.sum(exp_logits)
            idx = np.random.choice(range(len(self.agents)), p=probs)
        return self.agents[idx], idx

    def update_agent_info(self, idx, result):
        """
        Update the results of the agent based on the result of the game. (1 for win, -1 for loss, 0 for draw)
        Note that we use the negative of the result, because the result that is passed in belongs to the opposite agent.
        Maintains a running average of the negative results.
        """
        self.agent_num_plays[idx] += 1
        self.agent_results[idx] -= result

    def show_agents(self):
        performances = self.get_performance()
        sorted_idx = np.argsort(performances)[::-1]
        for i in sorted_idx:
            print(f"Agent {i}: {self.agent_names[i]}, Avg result: {performances[i]:.2f}, Play count: {self.agent_num_plays[i]}")
    
    def get_all_agents(self):
        return self.agents
    
    def get_performance(self):
        return self.agent_results / self.agent_num_plays

    def get_strongest_agent(self):
        strongest_idx = np.argmax(self.get_performance())
        return self.agents[strongest_idx], self.agent_names[strongest_idx]



