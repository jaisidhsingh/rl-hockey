import random
import pickle
import numpy as np
from collections import deque

class ReplayMemory:
    """
    A single replay buffer supporting both n-step returns and prioritized replay.
    
    When n_steps > 1, the transitions are aggregated using cumulative reward (discounted over n steps).
    When prioritized is True, the buffer assigns priorities to aggregated transitions and samples according
    to these priorities.
    
    The sample method returns:
      (state, action, reward, next_state, done, num_steps, indices, weights)
    
    When prioritized replay is disabled, indices is an array of sampled indices and weights is an array of ones.
    """
    def __init__(self, capacity, n_steps=1, gamma=0.99, seed=0, prioritized=False,
                 memory_alpha=0.6, memory_beta_start=0, memory_max_beta_step=100000):
        random.seed(seed)
        np.random.seed(seed)

        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.prioritized = prioritized

        # Main storage for aggregated transitions:
        # Each stored transition is a tuple:
        #   (state, action, cumulative_reward, final_next_state, final_done, num_steps)
        self.buffer = []
        self.position = 0
        
        # For n-step aggregation:
        self.n_step_buffer = deque()
        
        # Setup for prioritized replay:
        if self.prioritized:
            self.alpha = memory_alpha
            self.beta = memory_beta_start
            self.beta_start = memory_beta_start
            self.beta_step = 0
            self.max_beta_step = memory_max_beta_step
            # Priorities are kept in a numpy array of size `capacity`
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.max_priority = 1.0

    def _add(self, transition):
        # Add a single aggregated transition to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        if self.prioritized:
            self.priorities[self.position] = self.max_priority  # new samples get max priority
        self.position = (self.position + 1) % self.capacity

    def _get_n_step_info(self):
        """
        Compute n-step aggregated transition from the stored n-step buffer.
        Returns:
          (state, action, cumulative_reward, final_next_state, final_done, total_steps)
        """
        cumulative_reward = 0.0
        # Sum the discounted rewards until (a) n steps are reached or (b) a terminal is encountered
        for idx, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            cumulative_reward += (self.gamma ** idx) * reward
            if done:
                break
        state, action, _, _, _ = self.n_step_buffer[0]
        # Decide the final next_state and done flag:
        if len(self.n_step_buffer) >= self.n_steps:
            final_next_state, final_done = self.n_step_buffer[self.n_steps - 1][3], self.n_step_buffer[self.n_steps - 1][4]
        else:
            final_next_state, final_done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        return (state, action, cumulative_reward, final_next_state, final_done, idx + 1)

    def push(self, state, action, reward, next_state, done):
        """
        Push a new transition. First, add it to the n-step buffer.
        If enough transitions are collected, aggregate them and add the result.
        If the new transition is terminal and the buffer has less than n steps, flush it.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        # Flush the n-step buffer if not enough transitions and terminal reached
        if len(self.n_step_buffer) < self.n_steps and done:
            while self.n_step_buffer:
                aggregated = self._get_n_step_info()
                self._add(aggregated)
                self.n_step_buffer.popleft()
        # When enough transitions are accrued, aggregate the first n steps
        elif len(self.n_step_buffer) >= self.n_steps:
            aggregated = self._get_n_step_info()
            self._add(aggregated)
            self.n_step_buffer.popleft()

    def sample(self, batch_size):
        """
        Return:
          state, action, reward, next_state, done, num_steps, indices, weights
        If prioritized replay is disabled, indices are arbitrary and weights are ones.
        """
        if self.prioritized:
            valid_priorities = self.priorities[:len(self.buffer)]
            sample_probs = valid_priorities ** self.alpha
            sample_probs /= sample_probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
            samples = [self.buffer[idx] for idx in indices]
            
            # Update beta (for importance-sampling weights) linearly
            self.beta = min(1.0, self.beta_start + self.beta_step * (1.0 - self.beta_start) / self.max_beta_step)
            self.beta_step += 1
            
            weights = (len(self.buffer) * sample_probs[indices]) ** (-self.beta)
            weights /= weights.max()
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[idx] for idx in indices]
            weights = np.ones(batch_size, dtype=np.float32)
        
        # Unzip the sample tuple:
        state, action, reward, next_state, done, num_steps = map(np.stack, zip(*samples))
        return state, action, reward, next_state, done, num_steps, indices, weights

    def update_priorities(self, indices, priorities):
        """
        After training, update priorities for the sampled transitions. Only used if prioritized replay is enabled.
        """
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
                if priority > self.max_priority:
                    self.max_priority = priority

    def __len__(self):
        return len(self.buffer)
