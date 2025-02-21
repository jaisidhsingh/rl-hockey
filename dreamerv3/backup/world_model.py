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
        """
        Args:
            obs: [batch_size, *] or [batch_size, time, *]
        Returns:
            embed: [batch_size, embedding_size] or [batch_size, time, embedding_size]
        """
        # Get shapes
        batch_shape = obs.shape[:-1]  # [B] or [B, T]
        flat_shape = (-1, obs.shape[-1])  # [-1, obs_size]
        
        # Flatten batch and sequence dims
        obs = obs.reshape(flat_shape)
        
        # Transform
        obs = self.symlog.forward(obs)
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
    
    # Initialize state if None
    if state is None:
        state = self.rssm.initial(obs.shape[0])
    
    # RSSM forward pass
    next_state, states, features = self.rssm.observe(
        embed=embedded_obs,
        actions=actions,
        is_first=is_first,
        prev_state=state
    )
    
    # Extract tensor features for decoder
    features_tensor = self.rssm.get_feature(states)
    
    # Predict reconstructions, rewards, and discounts
    recon_obs = self.decode_obs(features_tensor)
    rewards = self.reward_predictor(features_tensor)
    discounts = self.discount_predictor(features_tensor)
    
    return next_state, {
        'states': states,
        'features': features_tensor,
        'recon_obs': recon_obs,
        'rewards': rewards,
        'discounts': discounts,
        'prior_logits': features['logit'],  # Add logits for KL loss
        'post_logits': features['logit']    # Both are same since we use same network
    }

   def imagine(self, state: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
       """Imagine forward using only actions."""
       states, features = self.rssm.imagine_forward(state, actions)
       rewards = self.reward_predictor(features)
       discounts = self.discount_predictor(features)
       return states, features, rewards, discounts

   def imagine_step(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Imagine one step into the future."""
        # Instead of: next_state = self.rssm.imagine(state, action)
        # Use forward directly:
        next_state, _ = self.rssm.forward(state, action, embed=None, is_first=None)
        
        # Extract features 
        features = self.rssm.get_feature(next_state)
        
        # Predict reward
        reward_logits = self.reward_predictor(features)  
        reward_probs = F.softmax(reward_logits, dim=-1)  
        
        # Create symexp-spaced bins
        bins = torch.linspace(-20, 20, 255, device=features.device)
        bins = torch.sign(bins) * (torch.exp(torch.abs(bins)) - 1)  
        
        # Compute expected reward
        reward = (reward_probs * bins).sum(dim=-1)  
        reward = reward * 0.1  
        
        # Predict discount
        discount_logits = self.discount_predictor(features)  
        discount = torch.sigmoid(discount_logits).squeeze(-1)  
        
        return next_state, reward, discount
        
   def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               discounts: torch.Tensor, is_first: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Compute world model losses."""
    print(f"Computing loss with shapes - obs: {obs.shape}, actions: {actions.shape}")
    
    # Forward pass
    _, model_outputs = self.forward(obs, actions, is_first)
    
    # Debug print
    print("Model outputs keys:", model_outputs.keys())
    print("States keys:", model_outputs['states'].keys())
    
    # Get predictions and reshape
    reward_pred = model_outputs['rewards']
    reward_pred = reward_pred.view(rewards.shape[0], rewards.shape[1], -1)
    reward_target = twohot_encode(rewards, num_bins=255)
    
    losses = {
        'world_model/prediction': (
            F.mse_loss(model_outputs['recon_obs'], obs) +
            F.cross_entropy(
                reward_pred.reshape(-1, 255),
                reward_target.reshape(-1, 255)
            ) +
            F.binary_cross_entropy_with_logits(
                model_outputs['discounts'].squeeze(-1),
                discounts
            )
        ),
        'world_model/dynamics': self.rssm.compute_dynamical_loss(
            model_outputs['post_logits'],  # Changed from states['logits']
            model_outputs['prior_logits']
        ).mean(),
        'world_model/representation': self.rssm.compute_kl_loss(
            model_outputs['post_logits'],  # Changed from states['logits']
            model_outputs['prior_logits']
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
    
    total_loss = (
        1.0 * losses['world_model/prediction'] +
        1.0 * losses['world_model/dynamics'] +
        0.1 * losses['world_model/representation']
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