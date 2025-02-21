from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from dataclasses import dataclass
import numpy as np
import wandb

from .rssm import SymlogTransform, ValueNormalizer, ReturnNormalizer
from .optimizer import LaProp, adaptive_gradient_clipping

@dataclass
class ActorCriticConfig:
   """Configuration for Actor and Critic."""
   # Architecture
   hidden_size: int = 1024
   layers: int = 3
   
   # Actor
   action_size: int = None
   action_discrete: bool = False
   action_hidden_size: int = 1024
   actor_grad_clip: float = 0.3  # AGC clip factor
   actor_entropy_scale: float = 3e-4  # η in paper
   actor_state_size: int = None
   
   # Critic
   critic_hidden_size: int = 1024
   critic_state_size: int = None
   critic_grad_clip: float = 0.3  # AGC clip factor
   critic_slow_target_fraction: float = 0.98  # From paper
   critic_slow_target_update: float = 1.0
   critic_bins: int = 255  # For two-hot value prediction
   
   # Training
   learning_rate: float = 3e-4
   eps: float = 1e-20  # For LaProp
   beta1: float = 0.9   # For LaProp
   beta2: float = 0.999 # For LaProp
   device: str = "cuda"
   agc_eps: float = 1e-3  # AGC epsilon
   
class TwoHotDistribution:
   """Two-hot encoded distribution for value prediction."""
   def __init__(self, logits: torch.Tensor, bins: int = 255):
        self.logits = logits
        self.bins = bins  # Store number of bins
        # Create symexp-spaced bins
        self.edges = torch.linspace(-20, 20, bins, device=logits.device)
        self.edges = torch.sign(self.edges) * (torch.exp(torch.abs(self.edges)) - 1)
    
       
   def mode(self) -> torch.Tensor:
       """Get most likely value."""
       probs = F.softmax(self.logits, dim=-1)
       return torch.sum(probs * self.edges, dim=-1)
   
   def mean(self) -> torch.Tensor:
       """Get distribution mean - same as mode for two-hot."""
       return self.mode()
   
   def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of value using two-hot encoding."""
        value_symlog = SymlogTransform.forward(value)
        below = torch.searchsorted(self.edges, value_symlog)
        above = below + 1
        
        # Fix clamp operations with self.bins
        below = torch.clamp(below, 0, self.bins - 1)
        above = torch.clamp(above, 0, self.bins - 1)
        
        # Distance-based weights with better numerical stability
        denom = self.edges[above] - self.edges[below]
        # Avoid division by zero
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        weight_above = (value_symlog - self.edges[below]) / denom
        weight_above = torch.clamp(weight_above, 0, 1)
        weight_below = 1 - weight_above
        
        # Create target distribution
        target = torch.zeros_like(self.logits)
        target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        
        # Cross entropy loss with better numerical stability
        log_probs = F.log_softmax(self.logits, dim=-1)
        return torch.sum(target * log_probs, dim=-1)

class ActionDistribution:
   """Distribution for continuous or discrete actions."""
   def __init__(self, logits: torch.Tensor, discrete: bool, unimix: float = 0.01):
       self.logits = logits
       self.discrete = discrete
       self.unimix = unimix
       
       if discrete:
           self.probs = F.softmax(logits, dim=-1)
           # Add uniform mixture for exploration
           self.mixed_probs = (1 - unimix) * self.probs + unimix / self.probs.shape[-1]
       else:
           # Split logits into mean and log_std for continuous actions
           mean, log_std = torch.chunk(logits, 2, dim=-1)
           self.mean = mean
           self.std = torch.exp(torch.clamp(log_std, -10, 2))
           self.dist = td.Normal(self.mean, self.std)
   
   def mode(self) -> torch.Tensor:
       """Get most likely action."""
       if self.discrete:
           return F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[-1])
       return torch.tanh(self.mean)
   
   def sample(self, amount: Optional[int] = None) -> torch.Tensor:
       """Sample action from distribution."""
       if self.discrete:
           if amount:
               return torch.multinomial(self.mixed_probs, amount)
           return torch.multinomial(self.mixed_probs, 1)
       sample = self.dist.rsample() if amount is None else self.dist.rsample((amount,))
       return torch.tanh(sample)
   
   def log_prob(self, action: torch.Tensor) -> torch.Tensor:
       """Compute log probability of action."""
       if self.discrete:
           return torch.sum(action * torch.log(self.mixed_probs + 1e-8), dim=-1)
       
       # Handle continuous actions with tanh transform
       action = torch.clamp(action, -0.999, 0.999)  # Avoid numerical issues
       log_prob = self.dist.log_prob(torch.atanh(action))
       return torch.sum(log_prob, dim=-1)
   
   def entropy(self) -> torch.Tensor:
       """Compute distribution entropy."""
       if self.discrete:
           return -torch.sum(self.mixed_probs * torch.log(self.mixed_probs + 1e-8), dim=-1)
       else:
           # For continuous actions, properly scale the entropy
           entropy = self.dist.entropy()
           # Account for tanh transform's effect on entropy
           entropy = entropy + np.log(2)  # Due to tanh squashing
           return entropy.sum(dim=-1)

class Actor(nn.Module):
   """Policy network with exploration and action space handling."""
   def __init__(self, config: ActorCriticConfig):
       super().__init__()
       self.config = config
       
       # Policy network
       output_size = config.action_size if config.action_discrete else config.action_size * 2
       self.policy = nn.Sequential(
           nn.Linear(config.actor_state_size, config.action_hidden_size),
           nn.SiLU(),
           nn.Linear(config.action_hidden_size, config.action_hidden_size),
           nn.SiLU(),
           nn.Linear(config.action_hidden_size, output_size),
       )
       
       # Zero init for last layer
       self.policy[-1].weight.data.zero_()
       self.policy[-1].bias.data.zero_()
       
       # Return normalization
       self.return_normalizer = ReturnNormalizer(config.device)
       
       # LaProp optimizer
       self.optimizer = LaProp(
           self.parameters(),
           lr=config.learning_rate,
           betas=(config.beta1, config.beta2),
           eps=config.eps
       )
   
   def forward(self, features: torch.Tensor) -> ActionDistribution:
       """Compute action distribution from features."""
       logits = self.policy(features)
       return ActionDistribution(logits, self.config.action_discrete)
   
   def get_actions(self, features: torch.Tensor, amount: Optional[int] = None) -> torch.Tensor:
       """Get actions for given features."""
       with torch.no_grad():
           dist = self.forward(features)
           return dist.sample(amount)
   
   def optimize(self, features: torch.Tensor, returns: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
       """Optimize actor using policy gradient with entropy regularization."""
       self.optimizer.zero_grad()
       
       # Get action distribution
       dist = self.forward(features)
       
       # Normalize returns using the return normalizer
       normalized_returns = self.return_normalizer.update(returns, training=True)
       
       # Compute loss terms
       log_probs = dist.log_prob(actions)
       entropy = dist.entropy()
       policy_loss = -(log_probs * normalized_returns.detach() + 
                    self.config.actor_entropy_scale * entropy)
       
       # Backward pass
       policy_loss.mean().backward()
       
       # Apply AGC
       clipped_gradients = adaptive_gradient_clipping(
           self.parameters(),
           {p: p.grad for p in self.parameters() if p.grad is not None},
           clip_factor=self.config.actor_grad_clip,
           eps=self.config.agc_eps
       )
       
       # Update gradients with clipped values
       if clipped_gradients:  # Check if we got valid gradients
           for param, grad in clipped_gradients.items():
               param.grad = grad
               
       self.optimizer.step()
       
       return {
           'policy_loss': policy_loss.mean().item(),
           'entropy': entropy.mean().item(),
           'normalized_returns': normalized_returns.mean().item(),
       }

class Critic(nn.Module):
   """Value network with two-hot encoded predictions and slow target network."""
   def __init__(self, config: ActorCriticConfig):
       super().__init__()
       self.config = config
       
       # Value network
       self.value = nn.Sequential(
           nn.Linear(config.critic_state_size, config.critic_hidden_size),
           nn.SiLU(),
           nn.Linear(config.critic_hidden_size, config.critic_hidden_size),
           nn.SiLU(),
           nn.Linear(config.critic_hidden_size, config.critic_bins),
       )
       
       # Zero init for last layer
       self.value[-1].weight.data.zero_()
       self.value[-1].bias.data.zero_()
       
       # Target network
       self.target_value = nn.Sequential(
           nn.Linear(config.critic_state_size, config.critic_hidden_size),
           nn.SiLU(),
           nn.Linear(config.critic_hidden_size, config.critic_hidden_size),
           nn.SiLU(),
           nn.Linear(config.critic_hidden_size, config.critic_bins),
       )
       
       # Initialize target network
       for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
           target_param.data.copy_(param.data)
           target_param.requires_grad = False
           
       # Initialize output layer to zero (critical for stability)
       torch.nn.init.zeros_(self.value[-1].weight)
       torch.nn.init.zeros_(self.value[-1].bias)
       torch.nn.init.zeros_(self.target_value[-1].weight)
       torch.nn.init.zeros_(self.target_value[-1].bias)
       
       # Value normalization
       self.value_normalizer = ValueNormalizer(config.device)
       
       # LaProp optimizer
       self.optimizer = LaProp(
           self.parameters(),
           lr=config.learning_rate,
           betas=(config.beta1, config.beta2),
           eps=config.eps
       )
   
   def forward(self, features: torch.Tensor, target_network: bool = False) -> TwoHotDistribution:
       """Get value distribution."""
       network = self.target_value if target_network else self.value
       logits = network(features)
       return TwoHotDistribution(logits, self.config.critic_bins)
   
   def get_values(self, features: torch.Tensor, target_network: bool = False) -> torch.Tensor:
       """Get values for given features."""
       with torch.no_grad():
           dist = self.forward(features, target_network)
           return dist.mode()
   
   def optimize(self, features: torch.Tensor, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
       """Optimize critic using two-hot value prediction."""
       self.optimizer.zero_grad()
       
       # Normalize returns
       normalized_returns = self.value_normalizer.update(returns)
       
       # Get value distributions
       dist = self.forward(features)
       target_dist = self.forward(features, target_network=True)
       
       # Compute losses
       value_loss = -dist.log_prob(normalized_returns)
       target_loss = -dist.log_prob(target_dist.mode().detach())
       total_loss = (value_loss + 
                    self.config.critic_slow_target_update * target_loss).mean()
       
       # Backward pass
       total_loss.backward()
       
       # Apply AGC
       clipped_gradients = adaptive_gradient_clipping(
           self.parameters(),
           {p: p.grad for p in self.parameters() if p.grad is not None},
           clip_factor=self.config.critic_grad_clip,
           eps=self.config.agc_eps
       )
       
       # Update gradients
       for p, grad in clipped_gradients.items():
           p.grad = grad
           
       self.optimizer.step()
       
       return {
           'value_loss': value_loss.mean().item(),
           'target_loss': target_loss.mean().item(),
           'predicted_values': dist.mode().mean().item(),
           'target_values': target_dist.mode().mean().item(),
       }
   
   def update_target_network(self):
       tau = self.config.critic_slow_target_fraction  # 0.98 from paper
       for target_param, param in zip(self.target_value.parameters(), 
                                     self.value.parameters()):
           target_param.data.copy_(
               tau * target_param.data + (1 - tau) * param.data
           )