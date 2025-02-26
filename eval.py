import os
import torch
import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
import wandb

from sac.sac import *
from td3.td3 import *

from utils.utils import load_agent, play
from utils.agent_picker import AgentPicker

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def play_hockey(player1, player2, num_episodes=1000, render=False):
    rewards = []
    results = []
    for i in tqdm(range(num_episodes), desc="Playing hockey"):
        reward, result = play(player1, player2, render=render)
        rewards.append(reward)
        results.append(result)
    print(f"Player 1 win ratio: {np.mean(results)}")
    print(f"Player 1 avg reward: {np.mean(rewards)}")
    return np.mean(results), np.mean(rewards)


def test_all_agents(player, opponent_dir, num_episodes=1000, render=False, uniform=False):
    agent_picker = AgentPicker(opponent_dir)

    agent_picker.init_agents(player)

    for i in tqdm(range(num_episodes), desc="Playing against other agents:"):
        opponent, idx = agent_picker.pick_agent(uniform=uniform)
        reward, result = play(player, opponent, render=render)
        agent_picker.update_agent_info(idx, result)
    
    agent_picker.show_agents()


def evaluate_all_agents(agents_dir="agents/", num_episodes=100):
    """
    Evaluates all agents against each other and creates a heatmap of win ratios.
    Saves results to Weights & Biases.
    
    Args:
        agents_dir (str): Directory containing the agent files
        num_episodes (int): Number of episodes to run for each agent pair
    """
    
    # Initialize wandb
    wandb.login(key="06c432da22d5e9e35fddc4c3d5febab30de45a02", verify=True)
    wandb.init(
        project="rl-self-play",
        entity="rl-project-2025",
        name="agent_evaluation",
    )

    # Get all .pt files from the directory
    agent_files = [f for f in os.listdir(agents_dir) if f.endswith('.pt')]
    n_agents = len(agent_files)
    
    # Create empty matrices for win rates and rewards
    win_matrix = np.zeros((n_agents, n_agents))
    reward_matrix = np.zeros((n_agents, n_agents))
    
    # Create a mapping of agents
    agents = {i: load_agent(os.path.join(agents_dir, fname)) 
             for i, fname in enumerate(agent_files)}
    
    # Evaluate all unique pairs
    for (i, j) in tqdm(list(combinations(range(n_agents), 2)), desc="Evaluating agent pairs"):
        win_rate, avg_reward = play_hockey(agents[i], agents[j], num_episodes=num_episodes)

        
        # Store results in both matrices (making them symmetric)
        win_matrix[i, j] = win_rate
        win_matrix[j, i] = -win_rate
        
        reward_matrix[i, j] = avg_reward
        reward_matrix[j, i] = -avg_reward
    
    print(win_matrix)
    # Calculate average win rate for each agent
    avg_win_rates = np.mean(win_matrix, axis=1)
    
    # Sort agents by average win rate
    sorted_indices = np.argsort(avg_win_rates)[::-1]
    
    # Reorder matrices and agent names
    win_matrix = win_matrix[sorted_indices][:, sorted_indices]
    reward_matrix = reward_matrix[sorted_indices][:, sorted_indices]
    agent_names = [agent_files[i].strip('.pt') for i in sorted_indices]
    
    # Create and log the plots to wandb
    plt.figure(figsize=(12, 5))
    
    # Win rate heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(win_matrix, annot=True, fmt='.2f', cmap='RdYlBu',
                xticklabels=agent_names, yticklabels=agent_names)
    plt.title('Agent Performance (row vs column)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Average reward heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(reward_matrix, annot=True, fmt='.2f', cmap='RdYlBu',
                xticklabels=agent_names, yticklabels=agent_names)
    plt.title('Average Rewards (row vs column)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Log matrices and plot to wandb
    wandb.log({
        "win_matrix": wandb.Image(plt),
        "win_rates": wandb.Table(
            data=[[agent_names[i]] + [f"{win_matrix[i][j]:.3f}" for j in range(n_agents)]
                 for i in range(n_agents)],
            columns=["Agent"] + agent_names
        ),
        "reward_matrix": wandb.Table(
            data=[[agent_names[i]] + [f"{reward_matrix[i][j]:.3f}" for j in range(n_agents)]
                 for i in range(n_agents)],
            columns=["Agent"] + agent_names
        ),
        "agent_ranking": wandb.Table(
            data=[[i+1, agent, avg_win_rates[sorted_indices[i]]] 
                 for i, agent in enumerate(agent_names)],
            columns=["Rank", "Agent", "Average Win Rate"]
        )
    })
    
    plt.show()
    wandb.finish()
    
    return win_matrix, reward_matrix, agent_names


def main():
    agent1_path = "agents/TD3-SP-2.pt"
    agent2_path = "agents/SAC-SP-2.pt"

    player1 = load_agent(agent1_path)
    player2 = load_agent(agent2_path)

    play_hockey(player1, player2, num_episodes=100, render=True)


if __name__ == "__main__":
    main()