import wandb
import numpy as np
import torch
import argparse
import os
import hockey.hockey_env as h_env
from collections import defaultdict, Counter

from sac.sac import *
from td3.td3 import *

from utils.agent_picker import AgentPicker
from utils.utils import play, load_agent

def evaluate_policy(player, opponent, eval_episodes=10):
    """
    Evaluates the player against a fixed opponent for multiple episodes
    """
    total_rewards = []
    wins = 0
    
    for _ in range(eval_episodes):
        reward, result = play(player, opponent)
        total_rewards.append(reward)
        wins += (result == 1)
    
    avg_reward = np.mean(total_rewards)
    win_rate = wins / eval_episodes
    std_reward = np.std(total_rewards)
    
    return avg_reward, std_reward, win_rate

def self_play_train():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--max_timesteps', type=int, default=1_000_000,
                        help='Maximum number of training timesteps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Size of each training batch')
    parser.add_argument('--warmup_steps', type=int, default=25_000,
                        help='Number of steps for random action warmup')
    parser.add_argument('--eval_freq', type=int, default=2_500,
                        help='How often to evaluate the agent')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='Number of training updates per environment step')
    
    # Self-play parameters
    parser.add_argument('--agent_name', type=str, default='td3_agent',
                        help='Base name for saved agent checkpoints (default: td3_agent)')
    parser.add_argument('--save_opponent_interval', type=int, default=50_000,
                        help='How often to save opponent versions')
    parser.add_argument('--checkpoint_dir', type=str, default='agents/self_play_pool')
    parser.add_argument('--initial_checkpoint', type=str, required=True,
                        help='Path to initial TD3 agent checkpoint to start self-play from')
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize environment
    env = h_env.HockeyEnv()

    wandb.init(
        project="rl-self-play",
        entity="rl-project-2025",
        config={
            "algorithm": "TD3-SelfPlay",
            "batch_size": args.batch_size,
            "save_opponent_interval": args.save_opponent_interval
        }
    )

    # Load initial agent and create opponent pool
    player = load_agent(args.initial_checkpoint)
    opponent_pool = AgentPicker(args.checkpoint_dir)
    # Keep a static evaluation opponent
    eval_opponent = load_agent(args.initial_checkpoint)

    # Initialize memory
    memory = ReplayBuffer(
        max_size=1_000_000
    )

    # Initialize opponent pool and run initial evaluation
    opponent_pool.init_agents(player)  # This runs initial games to set baseline probabilities

    # Add tracking variables for statistics
    episode_lengths = defaultdict(list)  # Track lengths per opponent
    win_types = []  # Track how games end
    
    # Training Loop
    total_timesteps = 0
    episode_num = 0

    state, _ = env.reset()
    state_opponent = env.obs_agent_two()
    episode_reward = 0
    episode_timesteps = 0

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        # Select actions
        if t < args.warmup_steps:
            with torch.no_grad():
                action = player.select_action(state)
        else:
            action = player.select_action(state)
            
        # Select opponent action
        opponent, opponent_idx = opponent_pool.pick_agent()
        action_opponent = opponent.select_action(state_opponent)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(
            np.hstack([action, action_opponent])
        )
        done = terminated or truncated
        next_state_opponent = env.obs_agent_two()

        # Store transition
        memory.add(state, action, reward, next_state, done)

        state = next_state
        state_opponent = next_state_opponent
        episode_reward += reward
        total_timesteps += 1

        # Update parameters
        if len(memory) > args.batch_size and t >= args.warmup_steps:
            for _ in range(args.updates_per_step):
                player.train(memory, args.batch_size)

        # Log episode results and update opponent pool
        if done:
            # Update opponent pool with game result
            opponent_pool.update_agent_info(opponent_idx, info["winner"])
            
            # Track statistics
            opponent_name = opponent_pool.agent_names[opponent_idx]
            episode_lengths[opponent_name].append(episode_timesteps)
            
            # Track win type
            if terminated:
                if info["winner"] == 1:
                    win_types.append("win")
                elif info["winner"] == -1:
                    win_types.append("lose")
                else:  # winner == 0
                    win_types.append("draw")
            else:
                win_types.append("timeout")
            
            # Calculate statistics for logging
            recent_win_types = win_types[-100:]  # Last 100 games
            win_type_counts = Counter(recent_win_types)
            
            # Selection probabilities
            opponent_performances = opponent_pool.get_performance()

            wandb.log({
                "training/reward": episode_reward,
                "training/episode_length": episode_timesteps,
                "training/episode": episode_num,
                
                # Opponent pool metrics
                "opponent_pool/opponent_performance": opponent_performances[opponent_idx],
                "opponent_pool/min_performance": np.min(opponent_performances),
                "opponent_pool/max_performance": np.max(opponent_performances),
                "opponent_pool/std_performance": np.std(opponent_performances),
                "opponent_pool/num_opponents": len(opponent_pool.agents),

                # Game statistics
                "game_stats/win_types": {
                    f"{k}": v/len(recent_win_types) 
                    for k, v in win_type_counts.items()
                },
                
                # Learning progress
                "progress/games_played": episode_num,
            }, step=t+1)

            
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            # Reset for next episode
            state, _ = env.reset()
            state_opponent = env.obs_agent_two()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            
            strongest_opponent, _ = opponent_pool.get_strongest_agent()
            
            initial_avg_reward, initial_std_reward, initial_win_rate = evaluate_policy(player, eval_opponent)
            strongest_avg_reward, strongest_std_reward, strongest_win_rate = evaluate_policy(player, strongest_opponent)
            
            wandb.log({
                "eval/initial_avg_reward": initial_avg_reward,
                "eval/initial_std_reward": initial_std_reward, 
                "eval/initial_win_rate": initial_win_rate,
                "eval/strongest_avg_reward": strongest_avg_reward,
                "eval/strongest_std_reward": strongest_std_reward,
                "eval/strongest_win_rate": strongest_win_rate
            }, step=t+1)

            opponent_pool.show_agents()

        # Save new opponent version
        if (t + 1) % args.save_opponent_interval == 0:
            opponent_path = os.path.join(args.checkpoint_dir, f'{args.agent_name}_opponent_{t+1}.pt')
            torch.save(player, opponent_path)
            opponent_pool.add_agent(load_agent(opponent_path), f'{args.agent_name}_opponent_{t+1}.pt', player)
            
            # Drop weakest opponent if pool gets too large
            if len(opponent_pool.agents) > 10:
                dropped_name = opponent_pool.drop_weakest_agent()
                print(f"Dropped agent {dropped_name} from opponent pool")

    torch.save(player, os.path.join(args.checkpoint_dir, f'{args.agent_name}_final.pt'))
    wandb.finish()

if __name__ == "__main__":
    self_play_train()
