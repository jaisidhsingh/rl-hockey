import torch
import argparse
import numpy as np


class ReplayBuffer(object):
    def __init__(self, total_buffer_size, state_dim, action_dim, device):
        self.dtype = torch.float32
        self.total_buffer_size = total_buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.state_store = torch.zeros(self.combine_dims(total_buffer_size, state_dim), dtype=self.dtype)
        self.next_state_store = torch.zeros(self.combine_dims(total_buffer_size, state_dim), dtype=self.dtype)
        self.action_store = torch.zeros(self.combine_dims(total_buffer_size, action_dim), dtype=self.dtype)
        self.reward_store = torch.zeros(total_buffer_size, dtype=self.dtype)
        self.done_store = torch.zeros(total_buffer_size, dtype=self.dtype)

        self.pointer = 0
        self.current_size = 0


    def combine_dims(self, x, y):
        if np.isscalar(y):
            return (x, y)
        
        assert type(y) in [list, tuple]
        return (x, *y)


    def push(self, state, action, reward, next_state, done):
        self.state_store[self.pointer] = torch.from_numpy(state)
        self.action_store[self.pointer] = torch.from_numpy(action)
        self.reward_store[self.pointer] = reward
        self.next_state_store[self.pointer] = torch.from_numpy(next_state)
        self.done_store[self.pointer] = done
        self.current_size = min(self.current_size + 1, self.total_buffer_size)


    def pop(self, batch_size=32):
        indices = torch.randint(0, self.current_size, (batch_size,))
        batch = {
            "state": self.state_store[indices].to(self.device),
            "action": self.action_store[indices].to(self.device),
            "reward": self.reward_store[indices].to(self.device),
            "next_state": self.next_state_store[indices].to(self.device),
            "done": self.done_store[indices].to(self.device)
        }
        return batch


def setup_td3_args():
    parser = argparse.ArgumentParser()
    # Global settings
    parser.add_argument("--experiment-name", type=str, default="td3_0")
    parser.add_argument("--results-folder", type=str, default="./results")
    parser.add_argument("--checkpoint-folder", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--model-hidden-dims", type=str, default="512,512")
    # Hyper-params: training
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--policy-learning-rate", type=float, default=1e-3)
    parser.add_argument("--Q-learning-rate", type=float, default=1e-4)
    # Hyper-params: loss computation
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--target-noise-value", type=float, default=0.1)
    # Hyper-params: frequencies
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--update-point", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    # Hyper-params: limits
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--per-epoch-steps", type=int, default=4000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--max-episode-length", type=int, default=1000)
    parser.add_argument("--total-buffer-size", type=int, default=1000000)

    args = parser.parse_args()
    return args
