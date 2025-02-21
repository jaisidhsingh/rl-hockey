import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass 
class RSSMConfig:
    """Configuration for RSSM."""
    action_size: int
    hidden_size: int          # Base hidden size
    deterministic_size: int   # Recurrent state size (4x hidden)
    stochastic_size: int      # Number of latent variables (hidden/16)
    class_size: int          # Classes per latent (same as stochastic)
    device: str = "cuda"
    learning_rate: float = 4e-5
    unimix: float = 0.01     # 1% uniform mixture for robustness
    free_nats: float = 1.0   # Exactly 1.0 nat threshold from paper
    blocks: int = 8          # Number of parallel blocks
    
    def __post_init__(self):
        # Validate dimensions
        assert self.deterministic_size == self.hidden_size * 4, \
            "Deterministic size must be 4x hidden size"
        assert self.stochastic_size == self.hidden_size // 16, \
            "Stochastic size must be hidden_size/16"
        assert self.class_size == self.stochastic_size, \
            "Class size must match stochastic size"
        # Ensure dimensions work with block structure
        assert self.deterministic_size % self.blocks == 0, \
            f"Deterministic size {self.deterministic_size} must be divisible by {self.blocks} blocks"

class SymlogTransform:
    """Symmetric logarithmic transform that preserves signs.
    From paper: 'symlog function compresses the magnitudes of both large positive and negative values'
    """
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """Transform input using symlog: sign(x) * log(|x| + 1)"""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    @staticmethod
    def inverse(x: torch.Tensor) -> torch.Tensor:
        """Inverse of symlog transform: sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class ValueNormalizer:
    """Normalizes values using exponential moving average of mean and std.
    From paper: Uses percentile-based normalization for stability."""
    def __init__(self, device: str, momentum: float = 0.99, eps: float = 1e-8):
        self.device = device
        self.momentum = momentum
        self.eps = eps
        self.mean = torch.zeros((), device=device)
        self.std = torch.ones((), device=device)
        
    def stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current mean and standard deviation."""
        return self.mean, self.std
        
    def update(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Update statistics and normalize input.
        Args:
            x: Input tensor to normalize
            training: Whether to update statistics
        Returns:
            Normalized tensor
        """
        if not training:
            return (x - self.mean) / (self.std + self.eps)
            
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_std = (batch_var + self.eps).sqrt()
        
        # Update running statistics
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.std = self.momentum * self.std + (1 - self.momentum) * batch_std
        
        # Normalize batch
        return (x - batch_mean) / (batch_std + self.eps)

class ReturnNormalizer:
    """Normalizes returns for more stable training.
    From paper: Uses percentile-based normalization with clipping for stability."""
    def __init__(self, device: str, momentum: float = 0.99, limit: float = 1.0, 
                 percentile_low: float = 5, percentile_high: float = 95):
        self.device = device
        self.momentum = momentum
        self.limit = limit
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.scale = torch.ones((), device=device)
        
    def update(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Update statistics and normalize input using percentile-based scaling.
        
        From paper: "only scale down large return magnitudes but leave small 
        returns below the threshold of L=1 untouched"
        
        Args:
            x: Returns tensor to normalize
            training: Whether to update statistics
        Returns:
            Normalized returns tensor
        """
        if not training:
            return x / torch.maximum(torch.as_tensor(self.limit), self.scale)
            
        # Compute percentile range per batch
        low_percentile = torch.quantile(x, self.percentile_low / 100)
        high_percentile = torch.quantile(x, self.percentile_high / 100)
        batch_scale = high_percentile - low_percentile
        
        # Update running scale with momentum
        self.scale = (self.momentum * self.scale + 
                     (1 - self.momentum) * batch_scale)
        
        # Normalize using max between limit and scale
        effective_scale = torch.maximum(
            torch.as_tensor(self.limit, device=self.device), 
            self.scale
        )
        return x / effective_scale

class RSSM(nn.Module):
    """Recurrent State-Space Model with discrete representations."""
    def __init__(self, config: RSSMConfig):
        """Initialize RSSM model.
        
        Args:
            config: RSSMConfig object containing model dimensions
        """
        super().__init__()
        self.config = config
        
        # Extract dimensions from config
        self.action_size = config.action_size
        self.deterministic_size = config.deterministic_size
        self.stochastic_size = config.stochastic_size
        self.class_size = config.class_size
        self.hidden_size = config.hidden_size

        # Build core networks
        self._build_core_network()
        
        # Prior network (3 layers with SiLU)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deterministic_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.stochastic_size * self.class_size)
        )
        
        # Zero-initialize output layer
        self.prior_net[-1].weight.data.zero_()
        self.prior_net[-1].bias.data.zero_()
        
        # Block GRU setup
        block_size = self.deterministic_size // self.config.blocks
        self.gru_blocks = nn.ModuleList([
            nn.GRUCell(3 * self.hidden_size, block_size)
            for _ in range(self.config.blocks)
        ])
        
        # Block mixing layer
        self.block_input = nn.Linear(self.deterministic_size, self.deterministic_size)

    @property
    def state_size(self) -> Dict[str, Tuple[int, ...]]:
        """Return expected shapes of internal states."""
        return {
            'deter': (self.deterministic_size,),
            'stoch': (self.stochastic_size, self.class_size)
        }

    def initial(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Return zero initial states."""
        return {
            'deter': torch.zeros(batch_size, self.deterministic_size, 
                            device=self.config.device),
            'stoch': torch.zeros(batch_size, self.stochastic_size, self.class_size,
                            device=self.config.device)
        }

    def observe(self,
                prev_state: Dict[str, torch.Tensor],
                embed: torch.Tensor,
                actions: torch.Tensor,
                is_first: torch.Tensor,
                training: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass with inputs to get state and features.
        
        Args:
            prev_state: Previous model state
            embed: Encoded observations [B, T, E] or [B, E]
            actions: Action tensors [B, T, A] or [B, A]
            is_first: Reset indicators [B, T] or [B]
            training: Whether in training mode
            
        Returns:
            Tuple of (next state, state entries for buffer, features)
        """
        print(f"RSSM observe called with embed shape: {embed.shape}, actions shape: {actions.shape}")

        # Handle single timestep case (convert to sequence of length 1)
        if embed.ndim == 2:
            embed = embed.unsqueeze(1)  # [B, 1, E]
            actions = actions.unsqueeze(1)  # [B, 1, A]
            is_first = is_first.unsqueeze(1)  # [B, 1]
        
        # Forward state for whole sequence
        states = []
        priors = []
        posts = []
        state = prev_state
        
        # Loop through sequence
        for t in range(actions.shape[1]):
            # Forward step
            state, out = self.forward(
                state,
                actions[:, t],
                embed[:, t],
                is_first[:, t] if is_first is not None else None
            )
            
            # Collect outputs
            states.append(state)
            priors.append(out['prior'])
            posts.append(out['post'])
        
        # Stack time dimension
        states = {
            k: torch.stack([s[k] for s in states], 1)
            for k in states[0].keys()
        }
        priors = torch.stack(priors, 1)
        posts = torch.stack(posts, 1)
        
        # Get features for policy/value nets
        features = {
            'deter': states['deter'],
            'stoch': states['stoch'],
            'logit': posts
        }
        
        # For single timestep case, return final state and squeeze outputs
        if actions.shape[1] == 1:
            state = {k: v[:, -1] for k, v in states.items()}
            features = {k: v.squeeze(1) for k, v in features.items()}
        
        return state, states, features

    def forward(self, 
                state: Dict[str, torch.Tensor],
                action: torch.Tensor,
                embed: torch.Tensor,
                is_first: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Single step forward pass of the RSSM.
        
        Args:
            state: Previous state dictionary with 'deter' and 'stoch'
            action: Action tensor [B, A]
            embed: Encoded observation [B, E]
            is_first: Optional reset indicator [B]
            
        Returns:
            Tuple of (next state dict, outputs dict)
        """
        # Reset states if needed
        if is_first is not None:
            is_first = is_first.to(torch.bool)
            init_state = self.initial(len(is_first))
            for key in state:
                state[key] = torch.where(
                    is_first.unsqueeze(-1), 
                    init_state[key],
                    state[key]
                )
        
        # Compute deterministic state using GRU
        deter = self._core_step(
            prev_deter=state['deter'],
            prev_stoch=state['stoch'],
            action=action
        )
        
        # Compute prior (before seeing observation)
        prior = self._prior(deter)
        
        # Compute posterior (after seeing observation)
        deter_embed = torch.cat([deter, embed], dim=-1)
        if embed.shape[-1] != deter.shape[-1]:
            # Add projection if needed
            deter_embed = nn.Linear(deter_embed.shape[-1], deter.shape[-1]).to(deter.device)(deter_embed)
        post = self._prior(deter_embed)  # Use same network for posterior
        
        # Sample stochastic state from posterior
        stoch = self._categorical_with_unimix(post).sample()  # [B, S]
        stoch = F.one_hot(stoch, num_classes=self.class_size)  # [B, S, C]
        
        # Return new state and outputs
        next_state = {'deter': deter, 'stoch': stoch}
        outputs = {'prior': prior, 'post': post}
        
        return next_state, outputs

    def imagine(self,
                state: Dict[str, torch.Tensor],
                actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Imagine future states from current state and actions.
        
        Args:
            state: Initial state dict
            actions: Action sequence to imagine [B, T, A]
            
        Returns:
            Final state and imagined feature sequence
        """
        # Collect imagined states
        states = []
        
        # Loop through action sequence
        for t in range(actions.shape[1]):
            # Forward imagination step
            state = self.imagine_step(state, actions[:, t])
            states.append(state)
            
        # Stack time dimension
        states = {
            k: torch.stack([s[k] for s in states], 1)
            for k in states[0].keys()
        }
        
        return state, states

    def imagine_step(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Imagine one step into the future."""
        # Instead of calling rssm.imagine, use rssm.forward directly for single step
        next_state, _ = self.forward(state, action, embed=None, is_first=None)
        
        # Extract features and predict reward/discount
        features = self.rssm.get_feature(next_state)
        
        # Predict reward
        reward_logits = self.reward_predictor(features)  # [B, 255]
        reward_probs = F.softmax(reward_logits, dim=-1)  # [B, 255]
        
        # Create symexp-spaced bins
        bins = torch.linspace(-20, 20, 255, device=features.device)
        bins = torch.sign(bins) * (torch.exp(torch.abs(bins)) - 1)  # symexp transform
        
        # Compute expected reward
        reward = (reward_probs * bins).sum(dim=-1)  # [B]
        reward = reward * 0.1  # Match JAX scaling
        
        # Predict discount (continuation)
        discount_logits = self.discount_predictor(features)  # [B, 1]
        discount = torch.sigmoid(discount_logits).squeeze(-1)  # [B]
        
        return next_state, reward, discount

    def get_feature(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features for policy/value networks."""
        # Flatten stochastic state and concatenate with deterministic
        stoch_flat = state['stoch'].reshape(state['stoch'].shape[:-2] + 
                                        (-1,))
        return torch.cat([state['deter'], stoch_flat], -1)

    def get_feature_size(self) -> int:
        """Returns total size of features when flattened."""
        return self.deterministic_size + self.stochastic_size * self.class_size

    def get_imagination_starts(self, states: Dict[str, torch.Tensor], 
                         n_last: int) -> Dict[str, torch.Tensor]:
        """Extract starting states for imagination."""
        # Extract last n states and reshape to combine batch and time
        batch_size = len(states['deter'])
        starts = {
            k: v[:, -n_last:].reshape(batch_size * n_last, *v.shape[2:])
            for k, v in states.items()
        }
        return starts

    def compute_kl_loss(self, 
                        post_logits: torch.Tensor, 
                        prior_logits: torch.Tensor,
                        free_nats: Optional[float] = None) -> torch.Tensor:
        """Compute representation loss KL(posterior || prior).
        
        From paper: "trains the representations to become more predictable allowing 
        us to use a factorized dynamics predictor"
        
        Args:
            post_logits: Posterior logits from encoder [B, T, stoch_size, class_size]
            prior_logits: Prior logits from dynamics [B, T, stoch_size, class_size]
            free_nats: Minimum KL value (defaults to config)
        """
        free_nats = free_nats or self.config.free_nats
        
        # Categorical distributions with 1% uniform mixture for stability
        post_dist = self._categorical_with_unimix(post_logits)
        prior_dist = self._categorical_with_unimix(prior_logits)
        
        # Compute KL divergence analytically for categorical
        kl = td.kl_divergence(post_dist, prior_dist)
        
        # Average over stochastic variables, keep batch and time dims
        kl = kl.mean(dim=-2)  # [-2] is stochastic dim
        
        if free_nats:
            # "Clipping the dynamics and representation losses below 1 nat"
            kl = torch.maximum(kl, torch.tensor(free_nats, device=kl.device))
            
        return kl

    def compute_dynamical_loss(self, 
                             post_logits: torch.Tensor, 
                             prior_logits: torch.Tensor,
                             free_nats: Optional[float] = None) -> torch.Tensor:
        """Compute dynamics loss KL(stop_grad(posterior) || prior).
        
        From paper: "trains the sequence model to predict the next representation"
        
        Args:
            post_logits: Posterior logits [B, T, stoch_size, class_size]
            prior_logits: Prior logits [B, T, stoch_size, class_size] 
            free_nats: Minimum KL value (defaults to config)
        """
        free_nats = free_nats or self.config.free_nats
        
        # Stop gradient on posterior as per paper
        post_dist = self._categorical_with_unimix(post_logits.detach())
        prior_dist = self._categorical_with_unimix(prior_logits)
        
        # Compute KL divergence
        kl = td.kl_divergence(post_dist, prior_dist)
        
        # Average over stochastic variables, keep batch and time dims
        kl = kl.mean(dim=-2)  # [-2] is stochastic dim
        
        if free_nats:
            # "Clipping the dynamics and representation losses below 1 nat"
            kl = torch.maximum(kl, torch.tensor(free_nats, device=kl.device))
            
        return kl
        
    def _categorical_with_unimix(self, logits: torch.Tensor) -> td.Distribution:
        """Create categorical distribution with uniform mixture for stability.
        
        From paper: "mix with 1% uniform to prevent zero probabilities"
        """
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - self.config.unimix) * probs + self.config.unimix / probs.shape[-1]
        return td.Categorical(probs=probs)

    def _build_core_network(self) -> None:
        """Build core recurrent network with block-diagonal structure."""
        # Input preprocessors for deterministic, stochastic, and action
        self.deter_input = nn.Sequential(
            nn.Linear(self.deterministic_size, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size)
        )
        self.stoch_input = nn.Sequential(
            nn.Linear(self.stochastic_size * self.class_size, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size)
        )
        self.action_input = nn.Sequential(
            nn.Linear(self.action_size, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size)
        )

    def _core_step(self,
               prev_deter: torch.Tensor,
               prev_stoch: torch.Tensor,
               action: torch.Tensor) -> torch.Tensor:
        """Single step of core recurrent network."""
        # Normalize action magnitude
        action = action / torch.maximum(action.norm(dim=-1, keepdim=True),
                                    torch.ones_like(action))
        
        # Process inputs
        deter_input = self.deter_input(prev_deter)
        stoch_input = self.stoch_input(prev_stoch.reshape(prev_stoch.shape[:-2] + (-1,)))
        action_input = self.action_input(action)
        
        # Prepare block inputs by repeating and concatenating
        block_input = torch.cat([deter_input, stoch_input, action_input], -1)
        block_input = block_input.unsqueeze(-2).repeat(1, self.config.blocks, 1)
        
        # Process through GRU blocks
        deter_state = prev_deter.chunk(self.config.blocks, dim=-1)
        deter_state = [
            block(block_input[:, i], deter_state[i])
            for i, block in enumerate(self.gru_blocks)
        ]
        deter_state = torch.cat(deter_state, dim=-1)
        
        # Mix between blocks
        deter_state = self.block_input(deter_state)
        
        return deter_state

    def _prior(self, deter: torch.Tensor) -> torch.Tensor:
        """Compute prior distribution from deterministic state."""
        hidden = self.prior_net(deter)
        # Reshape to categorical logits
        return hidden.reshape(hidden.shape[:-1] + 
                            (self.stochastic_size, self.class_size))