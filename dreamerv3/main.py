import argparse
import wandb
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm  # Change to tqdm.auto
import time
import sys
import functools
print = functools.partial(print, flush=True)  # Force flushing of print statements

import hockey.hockey_env as h_env

from dreamerv3 import DreamerV3, DreamerV3Config

def get_model_size_config(size_name: str) -> dict:
    """Get model architecture parameters based on model size."""
    sizes = {
        "12M": {
            "hidden_size": 256,          # Base hidden size for all networks
            "recurrent_size": 1024,      # RSSM deterministic state size
            "latent_size": 16,           # Both stochastic_size and class_size
        },
        "25M": {
            "hidden_size": 384,
            "recurrent_size": 3072,
            "latent_size": 24,
        },
        "50M": {
            "hidden_size": 512,
            "recurrent_size": 4096,
            "latent_size": 32,
        },
        "100M": {"hidden_size": 768, "recurrent_size": 6144, "latent_size": 48},
        "200M": {"hidden_size": 1024, "recurrent_size": 8192, "latent_size": 64},
        "400M": {"hidden_size": 1536, "recurrent_size": 12288, "latent_size": 96},
    }
    assert size_name in sizes, f"Model size must be one of {list(sizes.keys())}"
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
    if torch.cuda.is_available():
        print(f"\nCUDA Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1e6:.2f}MB")
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
    model_config = get_model_size_config(args.model_size)
    
    # Create environments first
    print("Creating environments...")
    env = h_env.HockeyEnv_BasicOpponent()
    eval_env = h_env.HockeyEnv_BasicOpponent()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)
    
    # Verify CUDA setup
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using {device.upper()} device")
    
    # Now create config with environment info
    config = DreamerV3Config(
        # Required
        obs_shape=env.observation_space.shape,
        action_size=env.action_space.shape[0],
        action_discrete=False,
        
        # Architecture (from model_config)
        hidden_size=model_config["hidden_size"],
        rssm_state_size=model_config["recurrent_size"],
        stochastic_size=model_config["latent_size"],
        class_size=model_config["latent_size"],
        
        # Training
        sequence_length=64,    # From paper
        horizon_length=15,     # From paper
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        
        # Other parameters from paper
        unimix=0.01,         # 1% uniform mixture
        free_nats=1.0,       # Exactly 1.0 nat threshold
        beta_dyn=1.0,
        beta_rep=0.1,
    )
    
    # Initialize wandb
    print("Setting up wandb...")
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
            "device": device
        }
    )
    
    # Create agent
    print("Creating agent...")
    agent = DreamerV3(config).to(device)
    
    # Training loop setup
    total_steps = 0
    episode_reward = 0
    episode_steps = 0
    obs, _ = env.reset()
    episode_count = 0
    last_step_time = time.time()
    
    # Single clean progress bar
    progress = tqdm(
        total=args.max_steps,
        desc="Training",
        ncols=80,
        unit="steps",
        leave=True
    )
    
    while total_steps < args.max_steps:
        try:
            current_time = time.time()
            
            # Environment interaction
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and update counters
            agent.observe(obs, action, reward, done)
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Update progress bar every 50 steps
            if total_steps % 50 == 0:
                progress.set_postfix({
                    'ep_reward': f'{episode_reward:.1f}',
                    'buffer_size': len(agent.replay_buffer),
                }, refresh=True)
                progress.update(50)
            
            # Train agent when we have enough data
            if len(agent.replay_buffer) >= agent.replay_buffer.sequence_length:
                metrics = agent.update_parameters()
                
                # Log basic metrics every sequence length (64 steps)
                if total_steps % config.sequence_length == 0:
                    wandb.log({
                        'training/episode_reward': episode_reward,
                        'training/episode_length': episode_steps,
                        'training/buffer_size': len(agent.replay_buffer),
                        'training/episodes': episode_count,
                        'performance/steps_per_second': config.sequence_length / (time.time() - current_time),
                    }, step=total_steps)
                
                # Log detailed model metrics every batch update
                # (sequence_length * batch_size = 1024 steps)
                if total_steps % (config.sequence_length * config.batch_size) == 0:
                    wandb.log({
                        'world_model/total_loss': metrics.get('world_model/total', 0),
                        'world_model/prediction_loss': metrics.get('world_model/prediction', 0),
                        'world_model/dynamics_loss': metrics.get('world_model/dynamics', 0),
                        'world_model/representation_loss': metrics.get('world_model/representation', 0),
                        'actor/policy_loss': metrics.get('actor/policy_loss', 0),
                        'actor/entropy': metrics.get('actor/entropy', 0),
                        'critic/value_loss': metrics.get('critic/value_loss', 0),
                        **log_system_stats()
                    }, step=total_steps)
                
                last_step_time = current_time
            
            # Handle episode end silently
            if done:
                episode_count += 1
                obs, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                
            # Save checkpoints silently
            if total_steps % args.save_frequency == 0:
                checkpoint_path = Path("checkpoints") / f"dreamerv3_{total_steps}.pt"
                agent.save(checkpoint_path)
                
        except Exception as e:
            progress.close()
            wandb.finish()
            raise e
    
    progress.close()
    wandb.finish()
    env.close()
    eval_env.close()
    agent.save("dreamerv3_final.pt")

if __name__ == "__main__":
    main()