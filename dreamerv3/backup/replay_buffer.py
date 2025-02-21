from typing import Dict, Tuple, Optional, List
import torch
import numpy as np
from collections import deque
import threading

class ReplayBuffer:
    """
    Replay buffer with online queue and uniform sampling.
    Stores full episodes and samples sequences.
    """
    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        action_size: int,
        obs_shape: tuple,
        latent_state_size: Dict[str, tuple],
        num_envs: int = 1,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.num_envs = num_envs
        self.device = device
        
        # Initialize storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool_)
        self.episode_starts = np.zeros(capacity, dtype=np.bool_)
        
        # Initialize latent state storage
        self.latent_states = {
            key: np.zeros((capacity, *size), dtype=np.float32)
            for key, size in latent_state_size.items()
        }
        
        # Episode storage
        self.episode_borders = []  # List of (start_idx, end_idx) tuples
        
        # Online queue for recent experience
        self.online_queue = deque(maxlen=sequence_length * num_envs * 2)
        
        self.current_idx = 0
        self.current_size = 0
        self.current_episode_start = 0
        
        # Threading lock for thread-safe updates
        self.lock = threading.Lock()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        latent_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Add a single transition to the buffer."""
        with self.lock:
            # Store transition
            self.observations[self.current_idx] = obs
            self.actions[self.current_idx] = action
            self.rewards[self.current_idx] = reward
            self.terminals[self.current_idx] = terminal
            self.episode_starts[self.current_idx] = (self.current_idx == self.current_episode_start)
            
            # Store latent state if provided - Move to CPU first
            if latent_state is not None:
                for key, value in latent_state.items():
                    # Move to CPU and convert to numpy
                    self.latent_states[key][self.current_idx] = value.detach().cpu().numpy()
            
            # Add to online queue
            self.online_queue.append(self.current_idx)
            
            # Update episode tracking
            if terminal:
                self.episode_borders.append(
                    (self.current_episode_start, self.current_idx + 1)
                )
                self.current_episode_start = (self.current_idx + 1) % self.capacity
            
            # Update buffer stats
            self.current_idx = (self.current_idx + 1) % self.capacity
            self.current_size = min(self.current_size + 1, self.capacity)

    def update_latent_states(self, indices: np.ndarray, latent_states: Dict[str, np.ndarray]) -> None:
        """Update latent states for given indices with new values."""
        with self.lock:
            for key, value in latent_states.items():
                self.latent_states[key][indices] = value

    def _get_sequence_indices(self, start_idx: int) -> np.ndarray:
        """Get valid sequence indices starting from start_idx."""
        indices = []
        current_idx = start_idx
        
        # Find episode boundaries
        episode_start = start_idx
        episode_end = self.capacity
        
        for start, end in self.episode_borders:
            if start <= start_idx < end:
                episode_start = start
                episode_end = end
                break
        
        # Collect sequence indices
        for _ in range(self.sequence_length):
            if current_idx >= episode_end:
                break
            indices.append(current_idx)
            current_idx += 1
            
        return np.array(indices)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of sequence data from the replay buffer."""
        with self.lock:
            print(f"=== Starting sampling with batch_size={batch_size} ===")
            # Verify we have enough data for a sequence
            if self.current_size < self.sequence_length:
                raise ValueError(f"Not enough data in buffer (have {self.current_size}, need {self.sequence_length})")
                
            # Adjust batch size if we don't have enough sequences
            if batch_size > self.current_size // self.sequence_length:
                print(f"Adjusting batch size from {batch_size} to {max(1, self.current_size // self.sequence_length)}")
                batch_size = max(1, self.current_size // self.sequence_length)
                
            sequences = []  # Will hold sequence indices [B, T]
            
            # Get recent sequences from online queue first
            online_indices = list(self.online_queue)
            print(f"Online queue size: {len(online_indices)}")
            
            try:
                # Sample from online queue until we run out or fill batch
                print("Sampling from online queue...")
                while len(sequences) < batch_size and len(online_indices) >= self.sequence_length:
                    start_idx = online_indices.pop(0)
                    print(f"  Trying start_idx={start_idx}")
                    seq_indices = self._get_sequence_indices(start_idx)
                    print(f"  Got sequence of length {len(seq_indices)}")
                    
                    if len(seq_indices) == self.sequence_length:
                        sequences.append(seq_indices)
                        print(f"  Added sequence, now have {len(sequences)}/{batch_size}")
                        # Remove indices that would overlap with this sequence
                        online_indices = [i for i in online_indices if i >= seq_indices[-1]]
                
                print(f"After online sampling: {len(sequences)}/{batch_size} sequences")
                
                # Fill remaining sequences with uniform sampling
                print("Starting uniform sampling...")
                sample_attempts = 0
                max_attempts = batch_size * 10  # Prevent infinite loops
                
                while len(sequences) < batch_size and sample_attempts < max_attempts:
                    sample_attempts += 1
                    print(f"\nAttempt {sample_attempts}/{max_attempts}")
                    
                    # Handle case where no episodes are complete yet
                    if not self.episode_borders:
                        print("No complete episodes, sampling from available data")
                        max_start = max(0, self.current_size - self.sequence_length)
                        start_idx = np.random.randint(0, max_start + 1)
                    else:
                        # Sample random episode
                        episode_idx = np.random.randint(len(self.episode_borders))
                        episode_start, episode_end = self.episode_borders[episode_idx]
                        print(f"Sampled episode {episode_idx}: ({episode_start}, {episode_end})")
                        
                        # Sample start index from episode if possible
                        max_start = episode_end - self.sequence_length 
                        if max_start <= episode_start:
                            print("Episode too short, skipping")
                            continue  # Episode too short, try another
                            
                        start_idx = np.random.randint(episode_start, max_start)
                        print(f"Sampled start_idx={start_idx}")
                    
                    seq_indices = self._get_sequence_indices(start_idx)
                    if len(seq_indices) == self.sequence_length:
                        sequences.append(seq_indices)
                        print(f"Added sequence, now have {len(sequences)}/{batch_size}")
                
                if sample_attempts >= max_attempts:
                    print("WARNING: Hit maximum sampling attempts!")
                
                print("\nStacking sequences...")
                # Stack sequences and convert to tensor
                indices = np.stack(sequences)  # [B, T]
                
                print("Creating batch tensors...")
                # Create batch with all tensors on correct device
                batch = {
                    'observations': torch.as_tensor(
                        self.observations[indices], device=self.device
                    ),
                    'actions': torch.as_tensor(
                        self.actions[indices], device=self.device
                    ),
                    'rewards': torch.as_tensor(
                        self.rewards[indices], device=self.device
                    ),
                    'terminals': torch.as_tensor(
                        self.terminals[indices], device=self.device
                    ),
                    'episode_starts': torch.as_tensor(
                        self.episode_starts[indices], device=self.device
                    ),
                    'latent_states': {
                        key: torch.as_tensor(value[indices], device=self.device)
                        for key, value in self.latent_states.items()
                    }
                }
                
                print("=== Sampling completed successfully ===")
                return batch
                    
            except Exception as e:
                print(f"\nERROR in sample method:")
                print(f"Current size: {self.current_size}")
                print(f"Online queue size: {len(self.online_queue)}")
                print(f"Episode borders: {self.episode_borders}")
                print(f"Sequence length: {self.sequence_length}")
                print(f"Exception: {str(e)}")
                raise e

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.current_size