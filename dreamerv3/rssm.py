from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

@dataclass
class RSSMConfig:
    # Required
    action_size: int
    
    # Architecture (no defaults - controlled from main)
    deterministic_size: int
    stochastic_size: int
    class_size: int
    hidden_size: int
    blocks: int = 8
    layers: int = 3
    
    # Training
    sequence_length: int = 50
    imagination_horizon: int = 15
    batch_size: int = 16
    learning_rate: float = 4e-5
    device: str = "cuda"
    
    # Model parameters
    unimix: float = 0.01  # Added: Uniform mixture for exploration
    free_nats: float = 1.0  # Exactly 1.0 nat threshold as per paper
    beta_dyn: float = 0.5  # Added: Dynamics loss weight
    beta_rep: float = 0.1  # Added: Representation loss weight


class BlockDiagonalGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int = 8):
        super().__init__()
        assert hidden_size % num_blocks == 0, "Hidden size must be divisible by num_blocks"
        assert input_size % num_blocks == 0, "Input size must be divisible by num_blocks"
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.input_block_size = input_size // num_blocks
        
        # Create parameters for each block
        self.weight_ih = nn.ParameterList([
            nn.Parameter(torch.randn(3 * self.block_size, self.input_block_size) / np.sqrt(self.input_block_size))
            for _ in range(num_blocks)
        ])
        self.weight_hh = nn.ParameterList([
            nn.Parameter(torch.randn(3 * self.block_size, self.block_size) / np.sqrt(self.block_size))
            for _ in range(num_blocks)
        ])
        self.bias_ih = nn.ParameterList([
            nn.Parameter(torch.zeros(3 * self.block_size))
            for _ in range(num_blocks)
        ])
        self.bias_hh = nn.ParameterList([
            nn.Parameter(torch.zeros(3 * self.block_size))
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        gates = []
        
        # Process each block separately for better parallelization
        x_blocks = x.chunk(self.num_blocks, dim=-1)
        h_blocks = h.chunk(self.num_blocks, dim=-1)
        
        for b in range(self.num_blocks):
            # Calculate block gates more efficiently
            gates_b = (F.linear(x_blocks[b], self.weight_ih[b], self.bias_ih[b]) +
                      F.linear(h_blocks[b], self.weight_hh[b], self.bias_hh[b]))
            gates.append(gates_b)
        
        gates = torch.cat(gates, dim=-1)
        gates = gates.view(batch_size, self.num_blocks, 3, -1)
        
        # Apply gates with better numerical stability
        reset = torch.sigmoid(gates[..., 0, :])
        update = torch.sigmoid(gates[..., 1, :])
        candidate = torch.tanh(reset * gates[..., 2, :])
        
        h_new = (1 - update) * h.view(batch_size, self.num_blocks, -1) + \
                update * candidate
        
        return h_new.view(batch_size, -1)

class ValueNormalizer(nn.Module):
    """Normalizer for values using running statistics."""
    def __init__(self, device, clip: float = 1000.0, decay: float = 0.99):
        super().__init__()
        self.clip = clip
        self.decay = decay
        self.register_buffer('mean', torch.zeros(1, device=device))
        self.register_buffer('std', torch.ones(1, device=device))
        self.register_buffer('count', torch.zeros(1, device=device))

    def update(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update statistics and normalize x."""
        # Detach to avoid backprop through running stats
        mean, std = self.mean.detach(), self.std.detach()
        
        # Update stats
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # Update running statistics using Welford's algorithm
        self.mean = self.mean + delta * batch_count / total_count
        self.std = (1 - self.decay) * torch.sqrt(batch_var + 1e-6) + self.decay * self.std
        self.count = total_count
        
        # Normalize
        x_norm = (x - mean) / (std + 1e-6)
        return torch.clamp(x_norm, -self.clip, self.clip)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using current statistics."""
        return torch.clamp((x - self.mean) / (self.std + 1e-6), -self.clip, self.clip)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize x using current statistics."""
        return x * (self.std + 1e-6) + self.mean

class SymlogTransform:
    """Symmetric logarithmic transform."""
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    @staticmethod
    def inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

class ReturnNormalizer(nn.Module):
    """Normalizer for returns using percentile statistics."""
    def __init__(self, device, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer('scale', torch.ones(1, device=device))
    
    def get_scale(self, returns: torch.Tensor) -> torch.Tensor:
        """Get current scale based on return percentiles."""
        p95 = torch.quantile(returns, 0.95)
        p05 = torch.quantile(returns, 0.05)
        return p95 - p05
    
    def update(self, returns: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Update scale using percentiles and normalize returns."""
        if training:
            current_scale = self.get_scale(returns)
            # Update running scale with EMA
            self.scale = self.decay * self.scale + (1 - self.decay) * current_scale
        
        # Normalize returns using max(1,S)
        normalized = returns / torch.maximum(torch.ones_like(self.scale), self.scale)
        return normalized

class RSSM(nn.Module):
    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        
        # Core sizes from config
        self.action_size = config.action_size
        self.deterministic_size = config.deterministic_size
        self.stochastic_size = config.stochastic_size
        self.class_size = config.class_size
        self.hidden_size = config.hidden_size
        
        # Input sizes for networks
        self.prior_input_size = self.deterministic_size  # h_t for prior
        self.posterior_input_size = self.deterministic_size + self.hidden_size  # [h_t, embed_t] for posterior
        self.embed_size = self.hidden_size  # Size of observation embeddings
        
        # Action and state encoders
        self.action_encoder = nn.Linear(self.action_size, self.hidden_size)
        self.stoch_encoder = nn.Linear(
            self.stochastic_size * self.class_size,  # 16 * 16 = 256
            self.hidden_size  # 256
        )
        
        # Initialize encoders with proper scaling (as per paper)
        torch.nn.init.orthogonal_(self.action_encoder.weight, gain=1.0)
        torch.nn.init.orthogonal_(self.stoch_encoder.weight, gain=1.0)
        torch.nn.init.zeros_(self.action_encoder.bias)
        torch.nn.init.zeros_(self.stoch_encoder.bias)
        
        # GRU with block-diagonal structure for better scaling
        self.gru = BlockDiagonalGRU(
            input_size=self.hidden_size * 2,  # [action_embed, stoch_embed]
            hidden_size=self.deterministic_size,
            num_blocks=config.blocks
        )
        
        # Prior network (predicts z_t from h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.prior_input_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.stochastic_size * self.class_size)
        )
        
        # Posterior network (predicts z_t from h_t and x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.posterior_input_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.stochastic_size * self.class_size)
        )

        # Store configuration
        self.blocks = config.blocks
        self.device = config.device
        self.unimix = config.unimix
        self.free_nats = config.free_nats
        
        # Normalizers for values
        self.value_normalizer = ValueNormalizer(config.device)
        self.reward_normalizer = ValueNormalizer(config.device)
        self.debug = False  # Add debug flag to RSSM class

        # Add input size checks
        print(f"Debug - RSSM initialization:")
        print(f"  action_size: {self.action_size}")
        print(f"  hidden_size: {self.hidden_size}")
        print(f"  stochastic_size: {self.stochastic_size}")
        print(f"  class_size: {self.class_size}")

    def _normalize_action(self, action):
        """Normalize actions as in JAX implementation."""
        if action is None:
            return action
        mag = torch.abs(action).max(dim=-1, keepdim=True)[0]
        mag = torch.maximum(mag, torch.ones_like(mag))
        return action / mag.detach()  # detach for gradient stability

    def _gru_core(self, prev_state: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """GRU core computation matching JAX implementation."""
        print("\nDEBUG - GRU Core:")
        print(f"Input shapes:")
        print(f"  prev_state: {prev_state.shape}")
        print(f"  stoch (pre-reshape): {stoch.shape}")
        print(f"  action: {action.shape}")
        
        # Normalize action
        action = self._normalize_action(action)
        
        # Reshape stochastic state
        stoch = stoch.reshape(stoch.shape[0], -1)
        print(f"  stoch (post-reshape): {stoch.shape}")
        
        # Process inputs
        action_embed = self.action_encoder(action)
        stoch_embed = self.stoch_encoder(stoch)
        
        print(f"Embedding shapes:")
        print(f"  action_embed: {action_embed.shape}")
        print(f"  stoch_embed: {stoch_embed.shape}")
        
        if self.debug:
            print(f"Embeddings shapes:")
            print(f"  action_embed: {action_embed.shape}")
            print(f"  stoch_embed: {stoch_embed.shape}")
        
        # Ensure matching batch sizes
        assert action_embed.shape[0] == stoch_embed.shape[0], \
            f"Batch size mismatch: {action_embed.shape[0]} vs {stoch_embed.shape[0]}"
        
        # Concatenate in the right order (matching JAX)
        gru_input = torch.cat([action_embed, stoch_embed], dim=-1)
        
        # Forward through block-diagonal GRU
        next_state = self.gru(gru_input, prev_state)
        
        return next_state

    def initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
       """Initialize the recurrent state."""
       return {
           'deterministic': torch.zeros(batch_size, self.deterministic_size, device=self.device),
           'stochastic': torch.zeros(batch_size, self.stochastic_size, self.class_size, device=self.device),
           'logits': torch.zeros(batch_size, self.stochastic_size, self.class_size, device=self.device)
       }

    def get_feature(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
       """Convert state dict to feature tensor."""
       stoch = state['stochastic'].reshape(*state['stochastic'].shape[:-2], -1)
       return torch.cat([state['deterministic'], stoch], -1)

    def forward(self, prev_state: Dict[str, torch.Tensor], prev_action: torch.Tensor,
           embed: Optional[torch.Tensor] = None, is_first: Optional[torch.Tensor] = None):
        print("\nDEBUG - RSSM Forward:")
        print(f"Input state shapes:")
        print(f"  deterministic: {prev_state['deterministic'].shape}")
        print(f"  stochastic: {prev_state['stochastic'].shape}")
        print(f"  logits: {prev_state['logits'].shape}")
        print(f"  prev_action: {prev_action.shape}")
        if embed is not None:
            print(f"  embed: {embed.shape}")

        if is_first is not None:
            # Ensure is_first has correct shape for broadcasting
            is_first = is_first.reshape(is_first.shape[0], 1)  # [B, 1]
            
            # Create new initial state
            init_state = self.initial_state(prev_state['deterministic'].shape[0])
            
            # Handle each state component with proper broadcasting
            prev_state = {
                'deterministic': torch.where(
                    is_first.unsqueeze(-1),  # [B, 1, 1] for broadcasting
                    init_state['deterministic'],
                    prev_state['deterministic']
                ),
                'stochastic': torch.where(
                    is_first.unsqueeze(-1).unsqueeze(-1),  # [B, 1, 1, 1] for stochastic shape
                    init_state['stochastic'],
                    prev_state['stochastic']
                ),
                'logits': torch.where(
                    is_first.unsqueeze(-1).unsqueeze(-1),  # [B, 1, 1, 1] for logits shape
                    init_state['logits'],
                    prev_state['logits']
                )
            }

        # Deterministic state update
        deterministic = self._gru_core(
            prev_state=prev_state['deterministic'],
            stoch=prev_state['stochastic'],
            action=prev_action
        )

        # Computer prior and posterior (if embed provided)
        prior_logits = self.prior_net(deterministic)
        prior_logits = prior_logits.reshape(*prior_logits.shape[:-1], self.stochastic_size, self.class_size)
        
        # Mix with uniform for prior distribution
        prior_probs = F.softmax(prior_logits, -1)
        prior_probs = (1 - self.unimix) * prior_probs + self.unimix / self.class_size

        if embed is None:
            # Prior forward pass
            logits = prior_logits
        else:
            # Posterior forward pass
            posterior_input = torch.cat([deterministic, embed], -1)
            posterior_logits = self.posterior_net(posterior_input)
            posterior_logits = posterior_logits.reshape(
                *posterior_logits.shape[:-1], self.stochastic_size, self.class_size)
            
            # Mix with uniform for posterior distribution
            post_probs = F.softmax(posterior_logits, -1)
            post_probs = (1 - self.unimix) * post_probs + self.unimix / self.class_size
            logits = posterior_logits

        # Sample stochastic state
        stochastic = torch.zeros_like(prior_probs)
        indices = torch.argmax(prior_probs, dim=-1)
        stochastic.scatter_(-1, indices.unsqueeze(-1), 1.0)

        # Construct state dictionaries
        prior = {'deterministic': deterministic, 'stochastic': stochastic, 'logits': prior_logits}
        post = {'deterministic': deterministic, 'stochastic': stochastic, 'logits': logits}
        
        print("\nDEBUG - RSSM Forward - Detail:")
        print(f"prev_state['stochastic'] shape: {prev_state['stochastic'].shape}")
        print(f"prev_state['stochastic'] requires_grad: {prev_state['stochastic'].requires_grad}")
        print(f"prev_state memory layout: {prev_state['stochastic'].stride()}")
        return post, prior

    def observe(self, embed: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor,
           prev_state: Optional[Dict[str, torch.Tensor]] = None):
        print("\nDEBUG - RSSM Observe:")
        print(f"Input shapes:")
        print(f"  embed: {embed.shape}")
        print(f"  actions: {actions.shape}")
        print(f"  is_first: {is_first.shape}")
        if prev_state is not None:
            print(f"Initial state shapes:")
            print(f"  deterministic: {prev_state['deterministic'].shape}")
            print(f"  stochastic: {prev_state['stochastic'].shape}")
            print(f"  logits: {prev_state['logits'].shape}")

        batch_size = embed.shape[0]
        seq_len = embed.shape[1]
        
        # Just validate shapes match without enforcing specific values
        assert actions.shape[:2] == embed.shape[:2], \
            f"Action shape mismatch: {actions.shape[:2]} vs embed shape {embed.shape[:2]}"
        assert is_first.shape[:2] == embed.shape[:2], \
            f"is_first shape mismatch: {is_first.shape[:2]} vs embed shape {embed.shape[:2]}"
        
        if prev_state is None:
            prev_state = self.initial_state(batch_size)
        
        # Validate initial state
        for k, v in prev_state.items():
            expected_shape = (batch_size,) + self.state_size[k]
            assert v.shape == expected_shape, \
                f"State shape mismatch for {k}: {v.shape} vs expected {expected_shape}"
        
        states = []
        priors = []
        features = []
        
        state = prev_state
        for t in range(seq_len):
            # Forward step with shape validation
            post, prior = self.forward(
                state,
                actions[:, t],
                embed[:, t],
                is_first[:, t] if is_first is not None else None
            )
            
            states.append(post)
            priors.append(prior)
            features.append(self.get_feature(post))
            state = post
        
        # Stack along time dimension
        states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
        priors = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0].keys()}
        features = torch.stack(features, dim=1)
        
        return state, states, priors, features

    def imagine(self, prev_state: Dict[str, torch.Tensor], action: torch.Tensor) -> Dict[str, torch.Tensor]:
       """Imagine next state given previous state and action (prior only)."""
       post, prior = self.forward(prev_state, action)
       return prior

    def imagine_forward(
       self, 
       state: Dict[str, torch.Tensor],
       actions: torch.Tensor
   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
       batch_size = actions.shape[0]
       seq_len = actions.shape[1]
       
       states = []
       features = []
       
       for t in range(seq_len):
           # Imagine next state
           state = self.imagine(state, actions[:, t])
           
           # Store results
           states.append(state)
           features.append(self.get_feature(state))
           
       # Stack along time dimension
       states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
       features = torch.stack(features, dim=1)
       
       return states, features

    def compute_loss(self, embed, actions, is_first, prev_state=None):
       # Get model predictions
       state, states, priors, features = self.observe(embed, actions, is_first, prev_state)
       
       # KL losses
       kl_loss = self.compute_kl_loss(states['logits'], priors['logits'])
       dyn_loss = self.compute_dynamical_loss(states['logits'], priors['logits'])
       
       # Apply free nats
       if self.config.free_nats > 0:
           kl_loss = torch.maximum(kl_loss, torch.tensor(self.config.free_nats, device=kl_loss.device))
           dyn_loss = torch.maximum(dyn_loss, torch.tensor(self.config.free_nats, device=dyn_loss.device))
       
       # Scale losses
       losses = {
           'kl': self.config.beta_rep * kl_loss,
           'dyn': self.config.beta_dyn * dyn_loss,
       }
       
       return losses, state, states, features

    def compute_kl_loss(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence with free bits."""
        # Get probabilities
        post_probs = F.softmax(post_logits, -1)
        prior_probs = F.softmax(prior_logits, -1)
        
        # Add uniform mixture for numerical stability
        post_probs = (1 - self.config.unimix) * post_probs + self.config.unimix / self.config.class_size
        prior_probs = (1 - self.config.unimix) * prior_probs + self.config.unimix / self.config.class_size
        
        # Compute KL divergence
        kl = torch.sum(post_probs * (torch.log(post_probs) - torch.log(prior_probs)), dim=-1)
        
        # Apply free bits - critical for preventing collapse
        kl = torch.maximum(kl, torch.tensor(self.config.free_nats, device=kl.device))
        
        return kl

    def compute_dynamical_loss(self, post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        """Compute dynamics loss with free bits."""
        # Similar KL computation as above
        post_probs = F.softmax(post_logits, -1)
        prior_probs = F.softmax(prior_logits, -1)
        
        # Add uniform mixture
        post_probs = (1 - self.config.unimix) * post_probs + self.config.unimix / self.config.class_size
        prior_probs = (1 - self.config.unimix) * prior_probs + self.config.unimix / self.config.class_size
        
        # Compute KL with free bits
        kl = torch.sum(post_probs * (torch.log(post_probs) - torch.log(prior_probs)), dim=-1)
        kl = torch.maximum(kl, torch.tensor(self.config.free_nats, device=kl.device))
        
        return kl

    def truncate_sequence(
       self, 
       sequence: Dict[str, torch.Tensor]
   ) -> Dict[str, torch.Tensor]:
       """Truncate sequence to last timestep."""
       return {k: v[:, -1] for k, v in sequence.items()}

    def get_imagination_starts(
        self,
        states: Dict[str, torch.Tensor],
        n_last: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get states to start imagination from."""
        batch_size = states['deterministic'].shape[0]
        seq_len = states['deterministic'].shape[1]
        n_last = n_last or self.config.imagination_horizon
        
        # Take last n_last states without debug prints
        start_idx = max(0, seq_len - n_last)
        
        starts = {}
        for k, v in states.items():
            last_states = v[:, start_idx:]
            reshaped = last_states.reshape(-1, *v.shape[2:])
            starts[k] = reshaped
        
        return starts

    @property
    def state_size(self) -> Dict[str, Tuple[int, ...]]:
        """Return state size information."""
        return {
            'deterministic': (self.deterministic_size,),
            'stochastic': (self.stochastic_size, self.class_size),
            'logits': (self.stochastic_size, self.class_size)
        }

    def to(self, device):
       """Move the model to the specified device."""
       self.device = device
       return super().to(device)

    def imagine_trajectories(
        self, 
        state: Dict[str, torch.Tensor],
        actor: nn.Module,
        horizon: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Imagine trajectories using learned policy."""
        batch_size = state['deterministic'].shape[0]
        
        states = []
        actions = []
        features = []
        
        for t in range(horizon):
            # Get current features
            feat = self.get_feature(state)
            features.append(feat)
            
            # Sample action from policy
            action = actor(feat)
            actions.append(action)
            
            # Imagine next state
            state = self.imagine(state, action)
            states.append(state)
        
        # Stack along time dimension
        states = {k: torch.stack([s[k] for s in states], dim=1) for k in states[0].keys()}
        actions = torch.stack(actions, dim=1)
        features = torch.stack(features, dim=1)
        
        return states, actions, features