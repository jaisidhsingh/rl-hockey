from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import wandb

from .rssm import RSSM, RSSMConfig, ValueNormalizer, SymlogTransform
from .optimizer import LaProp  # Add this import

@dataclass
class WorldModelConfig:
    # Required
    obs_shape: tuple
    action_size: int
    rssm: RSSMConfig
    
    # Architecture (must match RSSM config)
    embedding_size: int
    encoder_hidden_size: int
    decoder_hidden_size: int

class VectorEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 1024, embedding_size: int = 1024):
        super().__init__()
        self.symlog = SymlogTransform()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Preserve batch and sequence dimensions
        batch_shape = obs.shape[:-1]  # [B, T]
        flat_shape = (-1, obs.shape[-1])  # [-1, obs_size]
        
        # Flatten batch and sequence dims
        obs = obs.reshape(flat_shape)
        
        # Apply transforms
        obs = self.symlog.forward(obs)
        
        # Forward through model
        embed = self.model(obs)
        
        # Restore batch and sequence dims
        embed = embed.reshape(*batch_shape, -1)
        
        return embed

class VectorDecoder(nn.Module):
   """Decoder for vector observations."""
   def __init__(self, output_size: int, hidden_size: int = 1024, input_size: int = 1024):
       super().__init__()
       self.model = nn.Sequential(
           nn.Linear(input_size, hidden_size),
           nn.SiLU(),
           nn.Linear(hidden_size, hidden_size),
           nn.SiLU(),
           nn.Linear(hidden_size, output_size),
       )
       self.symlog = SymlogTransform()
       
   def forward(self, features: torch.Tensor) -> torch.Tensor:
       decoded = self.model(features)
       # Apply inverse symlog transform
       return self.symlog.inverse(decoded)

class DensePredictor(nn.Module):
   """Single layer predictor for reward and continuation."""
   def __init__(self, input_size: int, dist: str = 'symlog_disc'):
       super().__init__()
       self.dist = dist
       self.model = nn.Sequential(
           nn.Linear(input_size, 1024),
           nn.SiLU(),
           nn.Linear(1024, 255 if dist == 'symlog_disc' else 1)
       )
       # Initialize last layer to zero for stability
       self.model[-1].weight.data.zero_()
       self.model[-1].bias.data.zero_()

   def forward(self, features: torch.Tensor) -> torch.Tensor:
       return self.model(features)

class WorldModel(nn.Module):
   def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        # Validate dimensions
        input_size = np.prod(config.obs_shape)
        
        self.config = config
        
        # Core RSSM model
        self.rssm = RSSM(config.rssm)
        
        # Get feature size for decoders
        rssm_feature_size = self.rssm.get_feature_size()
        
        # Vector observation encoder/decoder
        self.encoder = VectorEncoder(
            input_size=input_size,
            hidden_size=config.encoder_hidden_size,
            embedding_size=config.embedding_size,
        )
        
        self.decoder = VectorDecoder(
            output_size=input_size,
            hidden_size=config.decoder_hidden_size,
            input_size=rssm_feature_size
        )
        
        # Reward and continuation predictors
        self.reward_predictor = DensePredictor(
            input_size=rssm_feature_size,
            dist='symlog_disc'
        )
        
        self.discount_predictor = DensePredictor(
            input_size=rssm_feature_size,
            dist='binary'
        )
        
        # Add LaProp optimizer
        self.optimizer = LaProp(
            self.parameters(),
            lr=config.rssm.learning_rate,
            eps=1e-20  # From paper
        )

   def _extract_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
       """Extract feature tensor from state dictionary."""
       return self.rssm.get_feature(state)

   def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
       """Encode observations."""
       return self.encoder(obs.float())

   def decode_obs(self, features: torch.Tensor) -> torch.Tensor:
       """Decode observations from features."""
       return self.decoder(features)

   def forward(self, obs: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor, 
               state: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
       """
       Forward pass through the world model.
       Args:
           obs: Observations [batch, time, obs_shape]
           actions: Actions [batch, time, action_size]
           is_first: Episode start indicators [batch, time]
           state: Optional initial state
       """
       # Encode observations
       embedded_obs = self.encode_obs(obs)
       
       # RSSM forward pass
       next_state, states, priors, features = self.rssm.observe(
           embed=embedded_obs,
           actions=actions,
           is_first=is_first,
           prev_state=state
       )
       
       # Predict reconstructions, rewards, and discounts
       recon_obs = self.decode_obs(features)
       rewards = self.reward_predictor(features)
       discounts = self.discount_predictor(features)
       
       return next_state, {
           'states': states,
           'priors': priors,
           'features': features,
           'recon_obs': recon_obs,
           'rewards': rewards,
           'discounts': discounts
       }

   def imagine(self, state: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
       """Imagine forward using only actions."""
       states, features = self.rssm.imagine_forward(state, actions)
       rewards = self.reward_predictor(features)
       discounts = self.discount_predictor(features)
       return states, features, rewards, discounts

   def imagine_step(
       self,
       state: Dict[str, torch.Tensor],
       action: torch.Tensor
   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
       """Imagine one step into the future."""
       # Get next state from imagination
       next_state = self.rssm.imagine(state, action)
       
       # Extract features
       features = self.rssm.get_feature(next_state)
       
       # Predict reward and discount
       reward_logits = self.reward_predictor(features)  # [B, 255]
       reward_probs = F.softmax(reward_logits, dim=-1)  # [B, 255]
       
       # Create symexp-spaced bins
       bins = torch.linspace(-20, 20, 255, device=features.device)
       bins = torch.sign(bins) * (torch.exp(torch.abs(bins)) - 1)  # symexp transform
       
       # Compute expected reward
       reward = (reward_probs * bins).sum(dim=-1)  # [B]
       
       # Scale predicted rewards for better stability
       reward = reward * 0.1  # Match JAX scaling
       
       # Predict discount (continuation)
       discount_logits = self.discount_predictor(features)  # [B, 1]
       discount = torch.sigmoid(discount_logits).squeeze(-1)  # [B]
       
       return next_state, reward, discount

   def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                   discounts: torch.Tensor, is_first: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
       """Compute world model losses."""
       # Forward pass
       _, model_outputs = self.forward(obs, actions, is_first)
       
       # Get predictions and reshape
       reward_pred = model_outputs['rewards']  # [batch, time, 255]
       reward_pred = reward_pred.view(rewards.shape[0], rewards.shape[1], -1)  # [batch, time, bins]
       
       # Convert rewards to two-hot encoding
       reward_target = twohot_encode(rewards, num_bins=255)  # [batch, time, bins]
       
       losses = {
           'world_model/prediction': (
               F.mse_loss(model_outputs['recon_obs'], obs) +  # Decoder loss
               F.cross_entropy(
                   reward_pred.reshape(-1, 255),  # [batch*time, bins]
                   reward_target.reshape(-1, 255)  # [batch*time, bins]
               ) +  # Reward prediction loss
               F.binary_cross_entropy_with_logits(  # Continue predictor loss
                   model_outputs['discounts'].squeeze(-1), 
                   discounts
               )
           ),
           'world_model/dynamics': self.rssm.compute_dynamical_loss(
               model_outputs['states']['logits'],
               model_outputs['priors']['logits']
           ).mean(),
           'world_model/representation': self.rssm.compute_kl_loss(
               model_outputs['states']['logits'],
               model_outputs['priors']['logits']
           ).mean(),
           # Individual components for debugging
           'world_model/reconstruction': F.mse_loss(model_outputs['recon_obs'], obs),
           'world_model/reward_pred': F.cross_entropy(
               reward_pred.reshape(-1, 255),
               reward_target.reshape(-1, 255)
           ),
           'world_model/continue_pred': F.binary_cross_entropy_with_logits(
               model_outputs['discounts'].squeeze(-1), discounts
           )
       }
       
       # Scale losses according to paper
       total_loss = (
           1.0 * losses['world_model/prediction'] +     # β_pred = 1
           1.0 * losses['world_model/dynamics'] +       # β_dyn = 1
           0.1 * losses['world_model/representation']   # β_rep = 0.1
       )
       losses['world_model/total'] = total_loss
       
       return losses, model_outputs

   def flush_rewards(self) -> None:
       """Reset reward normalization statistics."""
       self.rssm.reward_normalizer = ValueNormalizer(self.config.device)

   def train(self, training: bool = True) -> None:
       """Set training mode."""
       super().train(training)
       self.rssm.train(training)

   def eval(self) -> None:
       """Set evaluation mode."""
       super().eval()
       self.rssm.eval()

   def to(self, device) -> nn.Module:
       """Move model to device."""
       super().to(device)
       self.rssm.to(device)
       return self

def twohot_encode(x: torch.Tensor, num_bins: int = 255, min_val: float = -20, max_val: float = 20) -> torch.Tensor:
    """Convert values to two-hot encoding using symlog-spaced bins."""
    # Create bin edges
    bins = torch.linspace(min_val, max_val, num_bins, device=x.device)
    bins = torch.sign(bins) * (torch.exp(torch.abs(bins)) - 1)  # symexp spacing
    
    # Find bin indices
    x_transformed = torch.sign(x) * torch.log(torch.abs(x) + 1)  # symlog transform
    bin_idx = torch.searchsorted(bins, x_transformed)
    
    # Compute weights for adjacent bins
    lower_idx = torch.clamp(bin_idx - 1, 0, num_bins - 1)
    upper_idx = torch.clamp(bin_idx, 0, num_bins - 1)
    
    # Linear interpolation weights
    distance = (x_transformed - bins[lower_idx]) / (bins[upper_idx] - bins[lower_idx] + 1e-6)
    lower_weight = 1 - distance
    upper_weight = distance
    
    # Create two-hot tensor
    twohot = torch.zeros(*x.shape, num_bins, device=x.device)
    twohot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    twohot.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    
    return twohot