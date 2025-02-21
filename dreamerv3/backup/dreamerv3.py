from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import pathlib
import wandb
import traceback  # Add at the top with other imports

from .rssm import RSSMConfig
from .world_model import WorldModel, WorldModelConfig 
from .actor_critic import Actor, Critic, ActorCriticConfig
from .replay_buffer import ReplayBuffer

@dataclass
class DreamerV3Config:
    # Required - no defaults
    obs_shape: tuple
    action_size: int
    hidden_size: int             
    rssm_state_size: int      
    stochastic_size: int        
    class_size: int
    
    # Optional - with defaults
    action_discrete: bool = False
    embedding_size: Optional[int] = None
    learning_rate: float = 4e-5
    batch_size: int = 16
    device: str = "cuda"
    replay_buffer_size: int = 1_000_000
    sequence_length: int = 64  # From paper
    horizon_length: int = 15   # From paper
    unimix: float = 0.01
    free_nats: float = 0.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    
    def __post_init__(self):
        if self.embedding_size is None:
            self.embedding_size = self.hidden_size
        
        # Create RSSM config with all required dimensions
        self.rssm = RSSMConfig(
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            deterministic_size=self.rssm_state_size,     # From constructor
            stochastic_size=self.stochastic_size,        # From constructor
            class_size=self.class_size,                  # From constructor
            device=self.device,
            learning_rate=self.learning_rate,
            unimix=self.unimix,
            free_nats=self.free_nats
        )
        
        # Calculate total feature size from RSSM dimensions
        feature_size = self.rssm.deterministic_size + (self.rssm.stochastic_size * self.rssm.class_size)
        
        # Create World Model config
        self.world_model = WorldModelConfig(
            obs_shape=self.obs_shape,
            action_size=self.action_size,
            rssm=self.rssm,
            embedding_size=self.hidden_size,
            encoder_hidden_size=self.hidden_size,
            decoder_hidden_size=self.hidden_size
        )
        
        # Create Actor Critic config with correct feature size
        self.actor_critic = ActorCriticConfig(
            action_size=self.action_size,
            action_discrete=self.action_discrete,
            actor_state_size=feature_size,    # Use computed feature size
            critic_state_size=feature_size,   # Use same feature size
            learning_rate=self.learning_rate,
            device=self.device
        )

class DreamerV3(nn.Module):
    """Main DreamerV3 agent implementing world model based reinforcement learning."""
    
    def __init__(self, config: DreamerV3Config):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Calculate expected feature size from RSSM
        feature_size = (config.rssm_state_size + 
                       config.stochastic_size * config.class_size)
        
        # Validate dimensions match for both actor and critic
        assert feature_size == config.actor_critic.actor_state_size, \
            f"Actor state size ({config.actor_critic.actor_state_size}) must match RSSM feature size ({feature_size})"
        assert feature_size == config.actor_critic.critic_state_size, \
            f"Critic state size ({config.actor_critic.critic_state_size}) must match RSSM feature size ({feature_size})"
        
        # Initialize components
        self.world_model = WorldModel(config.world_model).to(self.device)
        self.actor = Actor(config.actor_critic).to(self.device)
        self.critic = Critic(config.actor_critic).to(self.device)
        
        # Get latent state size from RSSM
        latent_state_size = {
            'deter': (config.rssm_state_size,),
            'stoch': (config.stochastic_size, config.class_size)
        }
        
        # Initialize replay buffer with correct parameters
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_size,
            sequence_length=config.sequence_length,
            action_size=config.action_size,
            obs_shape=config.obs_shape,
            latent_state_size=latent_state_size,  # Pass RSSM state shapes
            device=self.device
        )

        # Initialize training state
        self.training = True
        self.train_step = 0
        self.last_state = None
        self.last_action = None

    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        """Get initial state for new episode."""
        return self.world_model.rssm.initial(batch_size=1)  # Use the 'initial' method

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action given observation."""
        # Convert observation to tensor and ensure batch dimension
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)  # [1, obs_size]
        
        # Initialize state if needed
        if self.last_state is None:
            self.last_state = self.get_initial_state()
            self.last_action = torch.zeros((1, self.config.action_size), device=self.device)
        
        # Encode observation
        embed = self.world_model.encode_obs(obs)
        
        # Update state using world model
        # observe returns (next_state, states, features)
        next_state, states, features = self.world_model.rssm.observe(
            prev_state=self.last_state,
            embed=embed,
            actions=self.last_action,
            is_first=torch.ones(1, device=self.device, dtype=torch.bool)
        )
        
        # Get features and select action
        features = self.world_model.rssm.get_feature(next_state)
        action = self.actor.get_actions(features)
        
        # Update tracking
        self.last_state = next_state
        self.last_action = action
        
        return action.cpu().numpy()[0]

    def observe(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
       """Add observation to replay buffer."""
       # Store transition with current state
       latent_state = None
       if self.last_state is not None:
           # Detach and move to CPU before storing
           latent_state = {
               k: v.detach().cpu() for k, v in self.last_state.items()
           }
       
       self.replay_buffer.add(
           obs=obs,
           action=action,
           reward=reward,
           terminal=done,
           latent_state=latent_state
       )
       
       # Reset state and action if episode ended
       if done:
           self.last_state = None
           self.last_action = None

    def update_parameters(self, batch_size: Optional[int] = None) -> Dict[str, float]:
       """Perform single training step."""
       try:
           batch_size = batch_size or self.config.batch_size
           print(f"Starting update with batch_size {batch_size}")
           
           if len(self.replay_buffer) < self.replay_buffer.sequence_length:
               print("Buffer too small, skipping update")
               return {'world_model/kl': 0.0}
           
           print("Sampling batch...")
           batch = self.replay_buffer.sample(batch_size)
           print("Batch sampled successfully")
           
           # Train world model silently
           wm_losses, wm_states = self.world_model.compute_loss(
               obs=batch['observations'],
               actions=batch['actions'],
               rewards=batch['rewards'],
               discounts=1.0 - batch['terminals'].float(),
               is_first=batch['episode_starts']
           )
           
           wm_losses['world_model/total'].backward()
           self.world_model.optimizer.step()
           
           # Get imagination starts
           B = batch_size
           K = min(self.config.horizon_length, batch['actions'].shape[1])
           H = self.config.horizon_length
           
           init_states = self.world_model.rssm.get_imagination_starts(
               wm_states['states'],
               n_last=K
           )
           
           # Imagine trajectories silently
           with torch.no_grad():
               state = init_states
               imagined_states = []
               imagined_actions = []
               rewards = []
               values = []
               
               features = self.world_model.rssm.get_feature(state)
               
               for _ in range(H):
                   action_dist = self.actor(features)
                   action = action_dist.sample()
                   value = self.critic(features).mode().squeeze(-1)
                   values.append(value)
                   
                   next_state, reward, discount = self.world_model.imagine_step(state, action)
                   next_features = self.world_model.rssm.get_feature(next_state)
                   
                   imagined_states.append(next_features)
                   imagined_actions.append(action)
                   rewards.append(reward)
                   
                   state = next_state
                   features = next_features
               
               # Stack time dimension
               imagined_states = torch.stack(imagined_states, dim=1)
               imagined_actions = torch.stack(imagined_actions, dim=1)
               rewards = torch.stack(rewards, dim=1)
               values = torch.stack(values, dim=1)
               
               # Compute returns
               returns = compute_lambda_return(
                   rewards=rewards,
                   values=values,
                   discount=0.997,
                   lambda_=0.95
               )
           
           # Train actor and critic silently
           actor_metrics = self.actor.optimize(
               features=imagined_states.detach(),
               returns=returns.detach(),
               actions=imagined_actions.detach()
           )
           
           critic_metrics = self.critic.optimize(
               features=imagined_states.detach(),
               returns=returns.detach()
           )

           self.critic.update_target_network()  # Update every step
           
           return {
               **wm_losses,
               'actor/policy_loss': actor_metrics['policy_loss'],
               'actor/entropy': actor_metrics['entropy'],
               'critic/value_loss': critic_metrics['value_loss'],
               'critic/target_value': critic_metrics['target_values']
           }
               
       except Exception as e:
           print(f"Error in update_parameters:")
           print(f"Exception: {type(e).__name__}: {str(e)}")
           print("Traceback:")
           print(traceback.format_exc())
           raise

    def train(self, training: bool = True) -> None:
       """Set training mode."""
       self.training = training
       self.world_model.train(training)
       self.actor.train(training)
       self.critic.train(training)

    def eval(self) -> None:
       """Set evaluation mode."""
       self.train(False)

    def save(self, path: Union[str, pathlib.Path]) -> None:
       """Save model to path."""
       torch.save({
           'world_model': self.world_model.state_dict(),
           'actor': self.actor.state_dict(),
           'critic': self.critic.state_dict(),
           'config': self.config,
           'train_step': self.train_step
       }, path)

    def load(self, path: Union[str, pathlib.Path]) -> None:
       """Load model from path."""
       checkpoint = torch.load(path)
       self.world_model.load_state_dict(checkpoint['world_model'])
       self.actor.load_state_dict(checkpoint['actor'])
       self.critic.load_state_dict(checkpoint['critic'])
       self.train_step = checkpoint['train_step']

    def to(self, device: Union[str, torch.device]) -> 'DreamerV3':
       """Move agent to device."""
       super().to(device)
       self.device = device
       self.world_model.to(device)
       self.actor.to(device)
       self.critic.to(device)
       if self.last_state is not None:
           self.last_state = {k: v.to(device) for k, v in self.last_state.items()}
       if self.last_action is not None:
            self.last_action = self.last_action.to(device)
       return self

def compute_lambda_return(rewards, values, discount=0.997, lambda_=0.95):
    """More accurate lambda return calculation matching JAX impl."""
    next_values = torch.cat([values[:, 1:], values[:, -1:]], dim=1)
    inputs = rewards + discount * next_values * (1 - lambda_)
    last = values[:, -1]
    outputs = []
    for t in reversed(range(rewards.shape[1])):
        last = inputs[:, t] + discount * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1)
    return returns