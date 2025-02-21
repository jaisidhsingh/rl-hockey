import argparse
import wandb
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys
import functools
# Override the built-in print function to force flushing of print statements.
# This ensures that all print outputs are immediately visible, which is useful
# for real-time logging and debugging, especially when running long training loops.
print = functools.partial(print, flush=True)

import hockey.hockey_env as h_env

from dreamerv3 import DreamerV3, DreamerV3Config

def get_model_size_config(size_name: str) -> dict:
    """Get model architecture parameters based on model size.
    From paper Table 3:
    - Hidden size is the base dimension
    - Recurrent size is 4x hidden size
    - Latent size is hidden_size/16 (both stochastic_size and class_size)
    """
    sizes = {
        "12M": {
            "hidden_size": 256,          # Base hidden size
            "recurrent_size": 1024,      # 4x hidden size
            "latent_size": 16,           # hidden_size/16
        },
        "25M": {
            "hidden_size": 384,
            "recurrent_size": 1536,
            "latent_size": 24,
        },
        "50M": {
            "hidden_size": 512,
            "recurrent_size": 2048,
            "latent_size": 32,
        },
        "100M": {
            "hidden_size": 768,
            "recurrent_size": 3072,
            "latent_size": 48,
        },
        "200M": {
            "hidden_size": 1024,
            "recurrent_size": 4096,
            "latent_size": 64,
        },
        "400M": {
            "hidden_size": 1536,
            "recurrent_size": 6144,
        },
    }
    assert size_name in sizes, f"Invalid model size '{size_name}'. Model size must be one of {list(sizes.keys())}"
    return sizes[size_name]

def parse_args():
    parser = argparse.ArgumentParser()
    # Model configuration
    parser.add_argument('--model_size', type=str, default='50M',
                       choices=['12M', '25M', '50M', '100M', '200M', '400M'],
                       help='Model size from paper')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--max_steps', type=int, default=2_000_000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--eval_frequency', type=int, default=5000)
    parser.add_argument('--save_frequency', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def evaluate_policy(agent: DreamerV3, env, steps: int, num_episodes: int = 5):
    """Run evaluation episodes."""
    agent.eval()
    episode_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
    
    agent.train()
    return np.mean(episode_rewards), np.std(episode_rewards)

def log_system_stats():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        return {
            'system/gpu_memory_allocated_mb': memory_allocated,
            'system/gpu_memory_cached_mb': memory_cached,
        }
    return {}

def print_cuda_memory():
    """
    Print the current CUDA memory status including allocated, cached, and max allocated memory.
    This is useful for monitoring GPU memory usage during training.
    """
    if torch.cuda.is_available():
        print(f"\nCUDA Memory Status:")
class DebugTqdm(tqdm):
    """A custom tqdm progress bar that forces stdout flush to ensure prints are visible."""
    print(f"Cached: {torch.cuda.memory_reserved(0)/1e6:.2f}MB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated(0)/1e6:.2f}MB")

class DebugTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, file=sys.stdout)
    
    def update(self, n=1):
        super().update(n)
        # Force flush stdout to ensure prints are visible
        sys.stdout.flush()

def main():
    args = parse_args()
    
    # Create weights directory if it doesn't exist
    Path("weights").mkdir(parents=True, exist_ok=True)
    
    model_config = get_model_size_config(args.model_size)
    
    # Create environment
    env = h_env.HockeyEnv_BasicOpponent()

    print(env.observation_space.shape)
    print(env.action_space.shape[0])
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset(seed=args.seed)
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Create config matching environment dimensions
    config = DreamerV3Config(
        action_discrete=False,
        hidden_size=model_config["hidden_size"],
        latent_dim=model_config["latent_size"],
        num_classes=model_config["latent_size"],
        class_size=model_config["latent_size"],
        deter_dim=model_config["recurrent_size"],
        device=device,
        capacity=1_000_000,  # 1M transitions
    )

    # Create agent with correct initialization parameters
    agent = DreamerV3(
        obs_shape=env.observation_space.shape,  # (18,)
        action_dim=env.action_space.shape[0],  # 4 with keep_mode
        config=config
    )
    
    # Initialize wandb
    wandb.init(
        project="dreamerv3-hockey",
        config={
            "algorithm": "dreamerv3",
            "model_size": args.model_size,
            "hidden_size": model_config["hidden_size"],
            "recurrent_size": model_config["recurrent_size"],
            "latent_size": model_config["latent_size"],
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "device": device,
            "env": "hockey",
        }
    )
    
    # Training loop
    total_steps = 0
    episode_reward = 0
    obs, _ = env.reset()
    
    progress = tqdm(total=args.max_steps)
    
    while total_steps < args.max_steps:
        # Get action and step environment
        action = agent.act(obs)  # Remove symlog here since it's handled in encoder
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.store_transition(obs, action, reward, next_obs, done)
        
        episode_reward += reward
        total_steps += 1
        
        # Train if enough data
        if len(agent.replay_buffer) >= config.batch_size * config.sequence_length:
            metrics = agent.train()
            
            # Log metrics every 1000 steps
            if total_steps % 1000 == 0:
                wandb.log({
                    'reward': episode_reward,
                    'world_model_loss': metrics['world_loss'],
                    'actor_loss': metrics['actor_loss'],
                    'critic_loss': metrics['critic_loss'],
                    'steps': total_steps,
                })
        
        if done:
            obs, _ = env.reset() 
            episode_reward = 0
        else:
            obs = next_obs
            
        # Update progress
        progress.update(1)
        
        # Save periodically
        if total_steps % args.save_frequency == 0:
            agent.save_checkpoint(f"dreamer_hockey_{total_steps}.pt")
    
    progress.close()
    wandb.finish()
    env.close()

if __name__ == "__main__":
    main()