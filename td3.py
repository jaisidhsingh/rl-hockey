import wandb
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from collections import deque
from importlib import reload
import hockey.hockey_env as h_env
reload(h_env)
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.2,
        policy_freq=2
    ):
        self.device = device
        
        # Initialize actors
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Initialize critics
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                

def evaluate_policy(policy, eval_env, t, eval_episodes=10, render=False):
    """
    Runs policy for X episodes and returns average reward.
    A fixed seed is used for the eval environment.
    """
    avg_reward = 0.
    all_rewards = []
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = policy.select_action(np.array(state), noise=0.0)  # No exploration noise during evaluation
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                eval_env.render()
        
        avg_reward += episode_reward
        all_rewards.append(episode_reward)
    
    avg_reward = avg_reward / eval_episodes
    std_reward = np.std(all_rewards)

    wandb.log({"mean_reward": avg_reward, "std_reward": std_reward}, step=t)    
    return avg_reward, std_reward


def main():
    # Environment setup
    env = h_env.HockeyEnv_BasicOpponent()
    eval_env = h_env.HockeyEnv_BasicOpponent()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = 123456
    env.reset(seed=seed)
    eval_env.reset(seed=seed)  # Different seed for eval env
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent and replay buffer
    td3 = TD3(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer()
    
    # Training parameters
    max_timesteps = 2_000_000
    batch_size = 32
    warmup_steps = 25_000
    eval_freq = 5000  # Evaluate every 5000 steps
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    wandb.init(project="rl-hockey-td3", config={}, name="other_2M")
    state, _ = env.reset()
    done = False
    
    # Keep track of evaluation metrics
    eval_rewards = []
    eval_steps = []

    for t in range(max_timesteps):
        episode_timesteps += 1

        # Select action
        if t < warmup_steps:
            action = env.action_space.sample()
        else:
            action = td3.select_action(np.array(state))

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store data in replay buffer
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= warmup_steps:
            td3.train(replay_buffer, batch_size)

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            avg_reward, std_reward = evaluate_policy(td3, eval_env, t)
            eval_rewards.append(avg_reward)
            eval_steps.append(t+1)
            torch.save(td3, "td3_agent.pt")

        if done:
            wandb.log({"episode_num": episode_num+1, "reward": episode_reward}, step=t+1)
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Final evaluation
    print("\nFinal Evaluation:")
    evaluate_policy(td3, eval_env, max_timesteps, eval_episodes=20)  # More episodes for final evaluation
    torch.save(td3, "td3_agent.pt")
    
    env.close()
    eval_env.close()
    
    return eval_rewards, eval_steps

if __name__ == "__main__":
    eval_rewards, eval_steps = main()
