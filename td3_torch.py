import os
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from hockey.hockey_env import HockeyEnv, HockeyEnv_BasicOpponent

from actor_critic import *
from utils import ReplayBuffer, setup_td3_args


class TD3(object):
    def __init__(self, args, env, eval_env, main_ac, tgt_ac):
        self.args = args
        self.env = env
        self.eval_env = eval_env
        self.main_ac = main_ac
        self.tgt_ac = tgt_ac

        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape[0]
        self.action_limit = env.action_space.high[0]

        for p in tgt_ac.parameters():
            p.requires_grad = False


    def get_action_main(self, state):
        action = self.main_ac(state) + self.args.noise_scale * torch.randn(self.action_dim)
        return action.clamp(-self.action_limit, self.action_limit)

    
    def get_action_tgt(self, state):
        action = self.tgt_ac(state) + self.args.noise_scale * torch.randn(self.action_dim)
        return action.clamp(-self.action_limit, self.action_limit)


    def policy_loss(self, state):
        loss = self.main_ac.Q1(state, self.main_ac(state))
        return -1 * loss.mean()


    def get_target(self, batch):
        next_state = batch["next_state"].to(self.args.device)
        dones = batch["done"].to(self.args.device)
        next_action = self.get_action(next_state)

        min_q_value = min(self.main_ac.Q1(next_state, next_action), self.main_ac.Q2(next_state, next_action))
        return batch["reward"] + self.args.gamma * (1 - dones) * min_q_value


    def Qs_loss(self, batch):
        [s, a, r, ns, d] = [batch[k] for k in batch.keys()]

        q1, q2 = self.main_ac.Q1(s, a), self.main_ac.Q2(s, a)

        with torch.no_grad():
            tgt_policy = self.tgt_ac.policy(ns)

            target_noise = torch.randn_like(tgt_policy) * self.args.target_noise_value
            target_noise = torch.clamp(target_noise, -1 * self.args.noise_clip, self.args.noise_clip)
            next_action = tgt_policy + target_noise

            Q1_tgt_policy = self.tgt_ac.Q1(ns, next_action)
            Q2_tgt_policy = self.tgt_ac.Q2(ns, next_action)
            Q_final_tgt_policy = torch.min(Q1_tgt_policy, Q2_tgt_policy)
            return_value = r + self.args.gamma * (1 - d) * Q_final_tgt_policy

        Q1_loss = F.mse(q1, return_value)
        Q2_loss = F.mse(q2, return_value)
        Q_loss = Q1_loss + Q2_loss

        loss_log = {"Q1_loss": Q1_loss.item(), "Q2_loss": Q2_loss.item()}
        
        return Q_loss, loss_log

    
    def update_td3_agent(self, batch, t):
        self.main_ac.train()
        self.Q_optimizer.zero_grad()
        
        Q_loss, loss_log = self.Qs_loss(batch)
        Q_loss.backward()

        if t % self.args.policy_delay == 0:
            for p in self.Q_params:
                p.requires_grad = False
            
            self.policy_optimizer.zero_grad()
            loss_policy = self.policy_loss(batch)
            loss_policy.backward()
            self.policy_optimizer.step()

            for p in self.Q_params:
                p.requires_grad = True
            
            with torch.no_grad():
                for p, p_ in zip(self.main_ac.parameters(), self.tgt_ac.parameters()):
                    p_.data = p_.data * self.args.rho + p_.data + (1 - self.args.rho) * p.data
    

    def test_td3_agent(self):
        self.main_ac.eval()

        eval_returns = 0
        eval_time_alive = 0

        for j in range(self.args.num_eval_episodes):
            state = self.eval_env.reset()
            done = False
            episode_return = 0
            episode_length = 0

            while done == False and episode_length != self.args.max_episode_length:
                state, reward, done, _ = self.eval_env.step(self.get_action_main(state))
                episode_return += reward
                episode_length += 1
            
            eval_returns += episode_return
            eval_time_alive += episode_length
        
        eval_returns /= self.args.num_eval_episodes
        eval_time_alive /= self.args.num_eval_episodes
    

    def save_checkpoint(self, dump):
        save_path = os.path.join(self.args.checkpoint_folder, self.experiment_name + f"_ckpt_{dump['epoch']}.pt")
        torch.save(dump, save_path)


    def run_td3(self):
        torch.manual_seed(self.args.random_seed)
        np.random_seed(self.args.random_seed)

        buffer = ReplayBuffer(self.args.total_buffer_size, self.state_dim, self.action_dim, self.args.device)

        total_steps = self.args.per_epoch_steps * self.args.num_epochs
        state = self.env.reset()
        episode_return = 0
        episode_length = 0

        for t in range(total_steps):
            if t > self.args.start_steps:
                action = self.get_action_main(state)
            else:
                action = self.env.action_space.sample()
            
            next_state, reward, done, _ = self.env.step(action)
            episode_return += reward
            episode_length += 1

            done = False if episode_length == self.args.max_episode_length else done
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done or (episode_length == self.args.max_episode_length):
                state = self.env.reset()
                episode_return = 0
                episode_length = 0
            
            if t >= self.args.update_point and t % self.args.update_every == 0:
                for j in range(self.args.update_every):
                    batch = buffer.pop(self.args.batch_size)
                    self.update_td3_agent(batch, t)
                
            if (t+1) % self.args.per_epoch_steps == 0:
                epoch = (t+1) // self.args.per_epoch_steps

                if epoch % self.args.save_every == 0:
                    self.save_checkpoint({"agent": self, "env": self.env, "epoch": epoch})
                
                self.test_td3_agent()

            
if __name__ == "__main__":
    args = setup_td3_args()
    env = HockeyEnv()
    eval_env = HockeyEnv_BasicOpponent()
    main_ac = ActorCritic(args, env)
    tgt_ac = deepcopy(main_ac)

    td3_agent = TD3(args, env, eval_env, main_ac, tgt_ac)
    td3_agent.run_td3()
