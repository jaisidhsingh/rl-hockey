import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical
import warnings

warnings.simplefilter("ignore")

def preprocess(image):
    return (image / 255.0).astype(np.float32)

def quantize(image):
    return (image * 255).astype(np.uint8)

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def twohot_encode(x, bins):
    """Encode scalar x into twohot vector using given bins"""
    x = x.unsqueeze(-1)
    bins = bins.to(x.device)
    k = torch.sum(bins < x, dim=-1)
    k = torch.clamp(k, 0, len(bins)-2)
    
    lo = bins[k]
    hi = bins[k+1] 
    w_hi = (x - lo) / (hi - lo)
    w_lo = 1 - w_hi
    
    output = torch.zeros(*x.shape[:-1], len(bins), device=x.device)
    output.scatter_(-1, k.unsqueeze(-1), w_lo.unsqueeze(-1))
    output.scatter_(-1, (k+1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return output

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DreamerV3Config:
    def __init__(self, action_discrete, hidden_size, latent_dim, class_size, deter_dim, device='cuda', **kwargs):
        # Environment specs
        self.action_discrete = action_discrete  # False for hockey
        
        # Architecture 
        self.hidden_size = hidden_size  # Base hidden size
        self.class_size = class_size  # Same as stochastic_size
        self.embed_dim = hidden_size  # Same as hidden size
        self.latent_dim = latent_dim 
        self.deter_dim = deter_dim  # Same as rssm_state_size
        # Training
        self.sequence_length = 64  # From paper
        self.batch_size = 16  # From paper
        self.imagination_horizon = 15  # From paper
        self.lr = 4e-5  # From paper
        self.actor_lr = 4e-5
        self.critic_lr = 4e-5
        self.eps = 1e-20  # LaProp epsilon
        self.capacity = 1_000_000  # 1M transitions
        self.device = device
        
        # Loss scales
        self.kl_scale = 1.0  # KL loss scale
        self.free_bits = 1.0  # Free bits
        self.entropy_coef = 3e-4  # Actor entropy coefficient
        
        # Other parameters
        self.discount = 0.997  # Discount factor
        self.lambda_ = 0.95  # GAE lambda
        
        # Override with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

class ReplayBuffer:
    def __init__(self, config, device):
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.episodes = []
        self.current_episode = []
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.current_episode.append({
            "obs": obs,  # Remove quantize for vector inputs
            "action": act,
            "reward": rew,
            "next_obs": next_obs,
            "done": done,
        })
        if done:
            if len(self.current_episode) >= self.sequence_length:
                self.episodes.append(self.current_episode)
            self.current_episode = []
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample(self, n_batches):
        valid_episodes = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        if not valid_episodes:
            raise StopIteration
        for _ in range(n_batches):
            batch_obs, batch_actions, batch_rewards, batch_dones = [], [], [], []
            for _ in range(self.batch_size):
                ep = valid_episodes[np.random.randint(len(valid_episodes))]
                start_idx = np.random.randint(0, len(ep) - self.sequence_length + 1)
                seq = ep[start_idx : start_idx + self.sequence_length]
                # Remove preprocess for vector inputs
                obs_seq = [transition["obs"] for transition in seq]
                action_seq = [transition["action"] for transition in seq]
                reward_seq = [transition["reward"] for transition in seq]
                done_seq = [transition["done"] for transition in seq]
                batch_obs.append(np.array(obs_seq))
                batch_actions.append(np.array(action_seq))
                batch_rewards.append(np.array(reward_seq))
                batch_dones.append(np.array(done_seq))
            yield {
                "observation": torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device),
                "action": torch.tensor(np.array(batch_actions), dtype=torch.float32, device=self.device),  # Changed to float32
                "reward": torch.tensor(np.array(batch_rewards), dtype=torch.float32, device=self.device),
                "done": torch.tensor(np.array(batch_dones), dtype=torch.float32, device=self.device),
            }

    def __len__(self):
        return len(self.episodes)

class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.SiLU(),
            nn.Linear(400, 400),
            nn.SiLU(),
            nn.Linear(400, embed_dim),
            nn.SiLU(),
        )
        self.apply(init_weights)

    def forward(self, x):
        # Apply symlog to vector inputs
        x = symlog(x)
        return self.net(x)

class ObservationDecoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 400),
            nn.SiLU(),
            nn.Linear(400, 400), 
            nn.SiLU(),
            nn.Linear(400, output_dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class TransitionDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dist_type="regression"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 200), nn.ReLU(), nn.Linear(200, out_dim)
        )
        self.dist_type = dist_type
        self.apply(init_weights)

    def forward(self, features):
        return self.net(features)

class RSSM(nn.Module):
    def __init__(self, action_dim, latent_dim, num_classes, deter_dim, embed_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.deter_dim = deter_dim
        self.embed_dim = embed_dim
        # Update input dimension for continuous actions
        self.gru = nn.GRUCell(latent_dim * num_classes + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim + action_dim, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim * num_classes),
        )
        self.apply(init_weights)

    def init_state(self, batch_size, device):
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.latent_dim, self.num_classes, device=device)
        stoch[:, :, 0] = 1.0
        return (stoch, deter)

    def observe(self, embed_seq, action_seq, init_state):
        T, B = action_seq.shape[:2]
        priors, posteriors, features = [], [], []
        stoch, deter = init_state
        for t in range(T):
            stoch_flat = stoch.view(B, -1)
            x = torch.cat([stoch_flat, action_seq[t]], dim=-1)
            deter = self.gru(x, deter)
            prior_input = torch.cat([deter, action_seq[t]], dim=-1)
            prior_logits = self.prior_net(prior_input).view(B, self.latent_dim, self.num_classes)
            prior_dist = Categorical(logits=prior_logits)
            post_input = torch.cat([deter, embed_seq[t]], dim=-1)
            posterior_logits = self.posterior_net(post_input).view(B, self.latent_dim, self.num_classes)
            posterior_dist = Categorical(logits=posterior_logits)
            # Straight-through sampling from the categorical
            stoch = F.gumbel_softmax(posterior_logits, tau=1.0, hard=True)
            feature = torch.cat([deter, stoch.view(B, -1)], dim=-1)
            features.append(feature)
            priors.append(prior_dist)
            posteriors.append(posterior_dist)
        return (priors, posteriors), torch.stack(features, dim=0)

    def imagine(self, init_state, actor, horizon):
        stoch, deter = init_state
        features, actions = [], []
        B = deter.size(0)
        for _ in range(horizon):
            feature = torch.cat([deter, stoch.view(B, -1)], dim=-1)
            features.append(feature)
            action_dist = actor(feature)
            action = action_dist.sample()
            actions.append(action)
            action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            stoch_flat = stoch.view(B, -1)
            x = torch.cat([stoch_flat, action_onehot], dim=-1)
            deter = self.gru(x, deter)
            prior_input = torch.cat([deter, action_onehot], dim=-1)
            prior_logits = self.prior_net(prior_input).view(B, self.latent_dim, self.num_classes)
            stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
        return torch.stack(features, dim=0), torch.stack(actions, dim=0)

class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim, embed_dim, latent_dim, num_classes, deter_dim, obs_shape, lr=1e-4, eps=1e-8):
        super().__init__()
        # Change input channels to be the first dimension of obs_shape
        self.encoder = ObservationEncoder(obs_shape[0], embed_dim)
        self.rssm = RSSM(action_dim, latent_dim, num_classes, deter_dim, embed_dim)
        feature_dim = deter_dim + latent_dim * num_classes
        # Change obs_size to obs_shape[0] for vector inputs
        self.decoder = ObservationDecoder(feature_dim, obs_shape[0])
        self.reward_decoder = TransitionDecoder(feature_dim, 1)
        self.continue_decoder = TransitionDecoder(feature_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=eps)

    def observe(self, observations, actions):
        T, B = observations.shape[:2]
        obs_flat = observations.reshape(T * B, *observations.shape[2:])
        embed = self.encoder(obs_flat).view(T, B, -1)
        init_state = self.rssm.init_state(B, observations.device)
        (priors, posteriors), features = self.rssm.observe(embed, actions, init_state)
        feat_dim = features.size(-1)
        features_flat = features.view(T * B, feat_dim)
        recon = self.decoder(features_flat).view(T, B, *observations.shape[2:])
        reward_pred = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        return (priors, posteriors), features, recon, reward_pred, continue_pred

    def imagine(self, init_state, actor, horizon):
        features, actions = self.rssm.imagine(init_state, actor, horizon)
        T, B, feat_dim = features.shape
        features_flat = features.view(T * B, feat_dim)
        reward_pred = self.reward_decoder(features_flat)
        continue_pred = self.continue_decoder(features_flat)
        return features, actions, reward_pred, continue_pred

    def decode(self, features):
        return self.decoder(features)

class Actor(nn.Module):
    def __init__(self, feature_dim, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 400),
            nn.SiLU(),
            nn.Linear(400, 400),
            nn.SiLU(),
            nn.Linear(400, 2 * action_size)  # Mean and log_std
        )
        self.apply(init_weights)
        
    def forward(self, x):
        x = self.net(x)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 200), nn.ReLU(), nn.Linear(200, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class DreamerV3:
    def __init__(self, obs_shape, action_dim, config):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.config = config
        self.replay_buffer = ReplayBuffer(config, config.device)
        self.device = config.device
        
        # Create world model with proper obs_shape
        self.world_model = WorldModel(
            obs_shape[0],  # in_channels (18 for hockey)
            action_dim,    # action dimension (4 for hockey)
            config.embed_dim,
            config.latent_dim,
            config.class_size,
            config.deter_dim,
            obs_shape,     # pass full obs_shape
            lr=config.lr,
            eps=config.eps
        ).to(self.device)
        feature_dim = config.deter_dim + config.latent_dim * config.num_classes
        self.actor = Actor(feature_dim, action_dim).to(self.device)
        self.critic = Critic(feature_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.hidden_state = None

    def init_hidden_state(self):
        self.hidden_state = self.world_model.rssm.init_state(1, self.device)

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.hidden_state is None:
                self.init_hidden_state()
            stoch, deter = self.hidden_state
            feature = torch.cat([deter, stoch.view(1, -1)], dim=-1)
            action_dist = self.actor(feature)
            action = action_dist.sample()
            # Remove one-hot encoding, just use continuous action directly
            stoch_flat = stoch.view(1, -1)
            x = torch.cat([stoch_flat, action], dim=-1)
            deter = self.world_model.rssm.gru(x, deter)
            prior_input = torch.cat([deter, action], dim=-1)
            prior_logits = self.world_model.rssm.prior_net(prior_input).view(
                1, self.world_model.rssm.latent_dim, self.world_model.rssm.num_classes
            )
            stoch = F.gumbel_softmax(prior_logits, tau=1.0, hard=True)
            self.hidden_state = (stoch, deter)
        return action.cpu().numpy()[0]  # Return as numpy array

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.store(obs, action, reward, next_obs, done)

    def update_world_model(self, batch):
        obs = batch["observation"]
        # Remove one-hot encoding for actions
        actions = batch["action"].float()
        rewards = batch["reward"].unsqueeze(-1)
        (priors, posteriors), features, recon, reward_pred, terminal_pred = self.world_model.observe(obs, actions)
        recon_loss = F.mse_loss(recon, obs)
        reward_loss = F.mse_loss(symlog(reward_pred), symlog(rewards.reshape(-1, 1)))
        dones = batch["done"].unsqueeze(-1).permute(1, 0, 2)
        terminal_loss = F.binary_cross_entropy_with_logits(terminal_pred, dones.reshape(-1, 1))
        kl_loss = 0
        T = len(priors)
        for t in range(T):
            kl_t = torch.distributions.kl_divergence(posteriors[t], priors[t]).mean()
            # Ensure KL loss is at least free_bits (prevent over-regularization)
            kl_loss += torch.max(kl_t, torch.tensor(self.config.free_bits, device=self.device))
        kl_loss = kl_loss / T
        world_loss = recon_loss + reward_loss + terminal_loss + self.config.kl_scale * kl_loss
        self.world_model.optimizer.zero_grad()
        world_loss.backward()
        self.world_model.optimizer.step()
        return {
            "world_loss": world_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "terminal_loss": terminal_loss.item(),
            "kl_loss": kl_loss.item()
        }

    def update_actor_and_critic(self, init_state):
        horizon = self.config.imagination_horizon
        features, actions, rewards, terminals = self.world_model.imagine(init_state, self.actor, horizon)
        features = features.detach()  # detach world model gradients
        rewards = rewards.detach()
        terminals = terminals.detach()
        T, B, feat_dim = features.shape
        values = self.critic(features.reshape(-1, feat_dim)).reshape(T, B, -1)
        discounts = self.config.discount * (1 - torch.sigmoid(terminals))
        returns = []
        future_return = values[-1]
        for t in reversed(range(T)):
            future_return = rewards[t] + discounts[t] * future_return
            returns.insert(0, future_return)
        returns = torch.stack(returns, dim=0)
        critic_loss = F.mse_loss(symlog(values), symlog(returns.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        features_flat = features.reshape(-1, feat_dim)
        actions_flat = actions.reshape(-1)
        action_dist = self.actor(features_flat)
        log_probs = action_dist.log_prob(actions_flat).reshape(T, B, 1)
        entropy = action_dist.entropy().mean()
        advantages = returns.detach() - values.detach()
        actor_loss = -(log_probs * advantages + self.config.entropy_coef * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_entropy": entropy.item()
        }

    def train(self):
        losses = {"world_loss": 0, "recon_loss": 0, "reward_loss": 0,
                  "terminal_loss": 0, "kl_loss": 0,
                  "actor_loss": 0, "critic_loss": 0, "actor_entropy": 0}

        batch = next(iter(self.replay_buffer.sample(1)))

        wm_losses = self.update_world_model(batch)
        for key in ["world_loss", "recon_loss", "reward_loss", "terminal_loss", "kl_loss"]:
            losses[key] += wm_losses[key]
        B = batch["observation"].shape[0]
        init_state = self.world_model.rssm.init_state(B, self.device)
        ac_losses = self.update_actor_and_critic(init_state)
        for key in ["actor_loss", "critic_loss", "actor_entropy"]:
            losses[key] += ac_losses[key]

        return losses
    
    def save_checkpoint(self, env_name):
        torch.save(self.world_model.state_dict(), f"weights/{env_name}_world_model.pt")
        torch.save(self.actor.state_dict(), f"weights/{env_name}_actor.pt")
        torch.save(self.critic.state_dict(), f"weights/{env_name}_critic.pt")

    def load_checkpoint(self, env_name):
        self.world_model.load_state_dict(
            torch.load(f"weights/{env_name}_world_model.pt")
        )
        self.actor.load_state_dict(torch.load(f"weights/{env_name}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"weights/{env_name}_critic.pt"))

def train_dreamer(args):
    env = AtariEnv(args.env).make()
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.n
    save_prefix = args.env.split("/")[-1]
    print(f"Env: {save_prefix}, Obs: {obs_shape}, Act: {act_dim}")
    
    config = Config(args)
    
    agent = DreamerV3(obs_shape, act_dim, config)
    agent.world_model.apply(init_weights)
    agent.actor.apply(init_weights)
    agent.critic.apply(init_weights)
    writer = SummaryWriter(log_dir=f"metrics/{save_prefix}")
    
    episode_history = []
    
    avg_reward_window = 50
    score, step = 0, 0
    state, _ = env.reset()
    agent.init_hidden_state()
    
    while len(episode_history) < config.episodes:
        action = agent.act(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.store_transition(state, action, reward, next_state, done)
        score += reward
        
        step += 1
        
        if done:
            ep = len(episode_history)
            episode_history.append(score)
            if len(agent.replay_buffer) > config.min_buffer_size:
                losses = agent.train()
                writer.add_scalar("Loss/World", losses["world_loss"], ep)
                writer.add_scalar("Loss/Recon", losses["recon_loss"], ep)
                writer.add_scalar("Loss/Reward", losses["reward_loss"], ep)
                writer.add_scalar("Loss/Terminal", losses["terminal_loss"], ep)
                writer.add_scalar("Loss/KL", losses["kl_loss"], ep)
                writer.add_scalar("Loss/Actor", losses["actor_loss"], ep)
                writer.add_scalar("Loss/Critic", losses["critic_loss"], ep)
                writer.add_scalar("Entropy/Actor", losses["actor_entropy"], ep)
            writer.add_scalar("Reward/Score", score, ep)
            avg_score = np.mean(episode_history[-avg_reward_window:])
            writer.add_scalar("Reward/Average", avg_score, ep)
            
            print(f"[Ep {ep:05d}/{config.episodes}] Score = {score:.2f} Avg.Score = {avg_score:.2f}", end="\r")
            
            if score >= max(episode_history, default=-np.inf):
                agent.save_checkpoint(save_prefix)

            score = 0
            agent.init_hidden_state()
            state, _ = env.reset()
        else:
            state = next_state

            
    
    print(f"\nFinished training. Final Avg.Score = {avg_score:.2f}")
    writer.close()