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
    parser.add_argument('--automatic_entropy_tuning', action='store_true',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--save_checkpoint_interval', type=int, default=200000, metavar='N',
                        help='Save a checkpoint per no. of steps (default: 200000)')
    parser.add_argument('--n_step_td', type=int, default=1, metavar='N',
                        help='Number of steps for TD update')
    parser.add_argument('--prioritized_replay', action='store_true',
                        help='Use prioritized experience replay when sampling from the buffer (default: False)')
    parser.add_argument('--agent_name', type=str, default='sac_agent',
                        help='Base name for saved agent checkpoints (default: sac_agent)')
    parser.add_argument('--max_timesteps', type=int, default=1_000_000, metavar='N',
                        help='Maximum number of training timesteps (default: 2,000,000)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='Batch size for training (default: 256)')
    parser.add_argument('--warmup_steps', type=int, default=25_000, metavar='N',
                        help='Number of warmup steps for exploration (default: 25,000)')
    parser.add_argument('--eval_freq', type=int, default=5000, metavar='N',
                        help='How often to evaluate the policy (default: 5,000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='Number of gradient updates per step (default: 1)')
    parser.add_argument('--memory_capacity', type=int, default=1_000_000, metavar='N',
                        help='Capacity of replay buffer (default: 1,000,000)')

    args = parser.parse_args()

    # Environment setup
    env = h_env.HockeyEnv_BasicOpponent()
    eval_env = h_env.HockeyEnv_BasicOpponent()
    
    # Set seeds
    seed = np.random.randint(1_000_000)
    env.reset(seed=seed)
    eval_env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize WandB
    wandb.login()
    wandb.init(
        project="rl-sac",
        config={
            "algorithm": "SAC",
            "policy": args.policy,
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": args.alpha,
            "automatic_entropy_tuning": args.automatic_entropy_tuning,
            "hidden_size": args.hidden_size,
            "learning_rate": args.lr,
            "n_step_td": args.n_step_td,
            "prioritized_replay": args.prioritized_replay,
            "batch_size": args.batch_size,
            "updates_per_step": args.updates_per_step,
            "max_timesteps": args.max_timesteps,
            "warmup_steps": args.warmup_steps,
            "eval_freq": args.eval_freq,
            "memory_capacity": args.memory_capacity
        }
    )

    # Initialize agent and memory with args
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayMemory(
        capacity=args.memory_capacity,
        seed=seed,
        n_steps=args.n_step_td,
        prioritized=args.prioritized_replay
    )

    # Training Loop
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    
    state, _ = env.reset()
    done = False

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        if t < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        total_timesteps += 1

        if len(memory) > args.batch_size and t >= args.warmup_steps:
            for i in range(args.updates_per_step):
                agent.update_parameters(memory, args.batch_size, t)

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_reward, std_reward = evaluate_policy(agent, eval_env, t)
            wandb.log({
                "reward/test": avg_reward,
                "reward/std": std_reward,
            }, step=t+1)

        if (t + 1) % args.save_checkpoint_interval == 0:
            torch.save(agent, f"agents/{args.agent_name}_{t + 1}.pt")

        if done:
            wandb.log({
                "reward/train": episode_reward,
                "episode_num": episode_num + 1,
                "episode_length": episode_timesteps
            }, step=t+1)
            
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    torch.save(agent, f"agents/{args.agent_name}_final.pt")
    
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    train()