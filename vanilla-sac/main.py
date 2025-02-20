import wandb
import gymnasium as gym
import numpy as np
import torch
import argparse
from importlib import reload
import hockey.hockey_env as h_env
reload(h_env)

from sac import SAC
from memory import ReplayMemory
# from utils import *

def evaluate_policy(policy, eval_env, t, eval_episodes=10, render=False):
    """
    Runs policy for X episodes and returns average reward.
    A fixed seed is used for the eval environment.
    """
    avg_reward = 0.
    all_rewards = []
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = policy.select_action(np.array(state), evaluate=True)  # No exploration during evaluation
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                eval_env.render()
        
        avg_reward += episode_reward
        all_rewards.append(episode_reward)
    
    avg_reward = avg_reward / eval_episodes
    std_reward = np.std(all_rewards)  
    
    return avg_reward, std_reward

def train():
    parser = argparse.ArgumentParser()
    # SAC specific parameters
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: True)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--save_checkpoint_interval', type=int, default=200000, metavar='N',
                        help='Save a checkpoint per no. of steps (default: 200000)')

    args = parser.parse_args()

    # Environment setup
    env = h_env.HockeyEnv_BasicOpponent()
    eval_env = h_env.HockeyEnv_BasicOpponent()
    
    # Set seeds
    seed = 123456
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Training parameters
    max_timesteps = 2_000_000
    # max_timesteps = 30_000
    batch_size = 256
    warmup_steps = 25_000
    eval_freq = 5000
    updates_per_step = 1
    
    # Initialize WandB
    wandb.login(key="06c432da22d5e9e35fddc4c3d5febab30de45a02", verify=True)
    wandb.init(
        project="rl-training",
        config={
            "algorithm": "SAC",
            "policy": args.policy,
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": args.alpha,
            "automatic_entropy_tuning": args.automatic_entropy_tuning,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "batch_size": batch_size,
            "updates_per_step": updates_per_step
        }
    )

    # Initialize agent and memory
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayMemory(1000000, seed)

    # Training Loop
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    
    state, _ = env.reset()
    done = False

    for t in range(max_timesteps):
        episode_timesteps += 1

        # Select action
        if t < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition in memory
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        total_timesteps += 1

        # Train agent after collecting sufficient data
        if len(memory) > batch_size and t >= warmup_steps:
            for i in range(updates_per_step):
                qf1, qf2, qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(memory, batch_size, t)
                # wandb.log({
                #     "stats/qf1": qf1,
                #     "stats/qf2": qf2,
                #     "stats/alpha": alpha,
                #     "losses/qf1_loss": qf1_loss,
                #     "losses/qf2_loss": qf2_loss,
                #     "losses/policy_loss": policy_loss,
                #     "losses/alpha_loss": alpha_loss
                # }, step=t+1)

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            avg_reward, std_reward = evaluate_policy(agent, eval_env, t)
            wandb.log({
                "reward/test": avg_reward,
                "reward/std": std_reward,
                "episode_num": episode_num + 1,
            }, step=t+1)

        if (t + 1) % args.save_checkpoint_interval == 0:
            torch.save(agent, f"agents/vanilla_sac_agent_{t + 1}.pt")

        if done:
            wandb.log({
                "reward/train": episode_reward,
                "episode_num": episode_num + 1,
                "episode_length": episode_timesteps
            }, step=t+1)
            
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Final evaluation
    print("\nFinal Evaluation:")
    evaluate_policy(agent, eval_env, max_timesteps, eval_episodes=20)
    
    # Save final models
    torch.save(agent, "agents/vanilla_sac_agent_final.pt")
    
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    train() 