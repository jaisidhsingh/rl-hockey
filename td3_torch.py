import os
import wandb
import torch
import itertools
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
        
        self.policy_optimizer = torch.optim.Adam(main_ac.policy.parameters(), lr=args.policy_learning_rate)
        self.Q_params = itertools.chain(main_ac.Q_function_1.parameters(), main_ac.Q_function_2.parameters())
        self.Q_optimizer = torch.optim.Adam(self.Q_params, lr=args.Q_learning_rate)

        wandb.init(project="rl-hockey-td3", config=vars(self.args), name=self.args.experiment_name)


    def get_action_main(self, state, testing=False):
        if not testing:
            action = self.main_ac(state) + self.args.noise_scale * torch.randn(self.action_dim).float().to(self.args.device)
        else:
            action = self.main_ac(state)
        return action.clamp(-self.action_limit, self.action_limit)

    
    def get_action_tgt(self, state, ):
        action = self.tgt_ac(state) + self.args.noise_scale * torch.randn(self.action_dim).float().to(self.args.device)
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
            Q_final_tgt_policy = torch.minimum(Q1_tgt_policy, Q2_tgt_policy)
            return_value = r.unsqueeze(-1) + self.args.gamma * (1 - d.unsqueeze(-1)) * Q_final_tgt_policy

        Q1_loss = F.mse_loss(q1, return_value)
        Q2_loss = F.mse_loss(q2, return_value)
        Q_loss = Q1_loss + Q2_loss

        loss_log = {"Q1_loss": Q1_loss.item(), "Q2_loss": Q2_loss.item()}
        
        return Q_loss, loss_log

    
    def update_td3_agent(self, batch, t):
        self.main_ac.train()

        self.Q_optimizer.zero_grad()
        Q_loss, loss_log = self.Qs_loss(batch)
        Q_loss.backward()
        self.Q_optimizer.step()

        policy_loss_value = 0

        if t % self.args.policy_delay == 0:
            for p in self.Q_params:
                p.requires_grad = False
            
            self.policy_optimizer.zero_grad()
            loss_policy = self.policy_loss(batch["state"])
            policy_loss_value = loss_policy.item()
            loss_policy.backward()
            self.policy_optimizer.step()

            for p in self.Q_params:
                p.requires_grad = True
            
            with torch.no_grad():
                for p, p_ in zip(self.main_ac.parameters(), self.tgt_ac.parameters()):
                    p_.data.mul_(self.args.rho)
                    p_.data.add_((1 - self.args.rho) * p.data)
                        
        loss_log.update({"policy_loss": policy_loss_value})
        # print(f"Time: {t} -- {loss_log}")
        wandb.log(loss_log, step=t)


    @torch.no_grad()
    def test_td3_agent(self):
        self.main_ac.eval()

        eval_returns = 0
        eval_time_alive = 0

        for j in range(self.args.num_eval_episodes):
            state, _ = self.eval_env.reset()
            done = False
            episode_return = 0
            episode_length = 0

            while done == False and episode_length != self.args.max_episode_length:
                state = torch.from_numpy(state).float().to(self.args.device)
                state, reward, done, _, __ = self.eval_env.step(self.get_action_main(state, testing=True).cpu().numpy())
                episode_return += reward
                episode_length += 1
            
            eval_returns += episode_return
            eval_time_alive += episode_length
        
        eval_returns /= self.args.num_eval_episodes
        eval_time_alive /= self.args.num_eval_episodes
        eval_log = {
            "eval_avg_return": eval_returns,
            "eval_avg_episode_length": eval_time_alive
        }
        return eval_log
    

    def save_checkpoint(self, dump):
        save_path = os.path.join(self.args.checkpoint_folder, self.experiment_name + f"_ckpt_{dump['epoch']}.pt")
        torch.save(dump, save_path)


    def run_td3(self):
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)

        os.makedirs(self.args.results_folder, exist_ok=True)
        os.makedirs(self.args.checkpoint_folder, exist_ok=True)

        buffer = ReplayBuffer(self.args.total_buffer_size, self.state_dim, self.action_dim, self.args.device)

        total_steps = self.args.per_epoch_steps * self.args.num_epochs
        state, _ = self.env.reset()
        episode_return = 0
        episode_length = 0

        for t in range(total_steps):
            if t > self.args.start_steps:
                state = torch.from_numpy(state).float().to(self.args.device)
                action = self.get_action_main(state).detach().cpu().numpy()
                state = state.detach().cpu().numpy()
            else:
                action = self.env.action_space.sample()
            
            next_state, reward, done, _, __ = self.env.step(action)
            episode_return += reward
            episode_length += 1

            done = False if episode_length == self.args.max_episode_length else done
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if done or (episode_length == self.args.max_episode_length):
                state, _ = self.env.reset()
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
                
                eval_log = self.test_td3_agent()
                eval_log.update({"epoch": epoch})
                # print(f"Time: {t} -- {eval_log}")
                wandb.log(eval_log, step=t)

            
if __name__ == "__main__":
    args = setup_td3_args()
    env = HockeyEnv_BasicOpponent()
    eval_env = HockeyEnv_BasicOpponent()
    main_ac = ActorCritic(args, env).to(args.device)
    tgt_ac = deepcopy(main_ac)

    td3_agent = TD3(args, env, eval_env, main_ac, tgt_ac)
    td3_agent.run_td3()
