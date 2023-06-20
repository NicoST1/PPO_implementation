import slimevolleygym
import base_networks
import pandas as pd
import numpy as np
import torch
import gym
import os
import re


class PPO(object):
    def __init__(self, net_width, gamma, gae_lambda, learning_rate,
                 clip_rate, num_epochs, steps_rollout, evaluate=True, load_previous=False, save_interval=0):

        self.environment = gym.make("SlimeVolley-v0")
        self.action_table = {0: [0, 0, 0],  # NOOP
                             1: [1, 0, 0],  # LEFT (forward)
                             2: [1, 0, 1],  # UP-LEFT (forward jump)
                             3: [0, 0, 1],  # UP (jump)
                             4: [0, 1, 1],  # UPRIGHT (backward jump)
                             5: [0, 1, 0]}  # RIGHT (backward)

        if not os.path.exists('model'):
            os.mkdir('model')

        state_dim = self.environment.observation_space.shape[0]
        action_dim = len(self.action_table)

        self.actor = base_networks.Actor([state_dim, net_width, net_width, action_dim])
        self.critic = base_networks.Critic([state_dim, net_width, net_width, 1])

        steps = [int(n) for n in re.findall('[0-9]+', ''.join(os.listdir('model')))]

        if len(steps) == 0 or not load_previous:
            self.total_steps = 0
        else:
            self.total_steps = max(steps)
            self.load(self.total_steps)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.steps_rollout = steps_rollout

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_rate = clip_rate
        self.num_epochs = num_epochs

        self.evaluate = evaluate
        self.save_interval = save_interval

    def test(self, num_episodes, render=False):
        rewards_per_episode = []

        for _ in range(num_episodes):
            state, done, episode_reward = self.environment.reset(), False, 0

            while not done:
                action, _ = self.actor.get_best_action(state)
                state, reward, done, _ = self.environment.step(self.action_table[action])

                episode_reward += reward
                if render:
                    self.environment.render()

            rewards_per_episode.append(episode_reward)

        if render:
            self.environment.close()

        return np.mean(rewards_per_episode)

    def rollout(self):

        states = []
        actions = []
        rewards = []
        next_states = []
        action_probs = []
        done_flags = []

        steps = 0

        while steps < self.steps_rollout:
            state = self.environment.reset()
            done = False

            while not done and steps < self.steps_rollout:
                states.append(state)

                action, action_prob = self.actor.sample_action(state)
                state, reward, done, info = self.environment.step(self.action_table[action])

                actions.append([action])
                rewards.append([reward])
                next_states.append(state)
                action_probs.append([action_prob])
                done_flags.append([done])

                steps += 1
                self.total_steps += 1

                if (self.save_interval > 0) and (self.total_steps % self.save_interval == 0):
                    self.save(self.total_steps)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.int64)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        action_probs = torch.tensor(np.array(action_probs), dtype=torch.float)
        done_flags = torch.tensor(np.array(done_flags), dtype=torch.float)

        return states, actions, rewards, next_states, action_probs, done_flags

    def compute_advantages(self, states, next_states, rewards, done_flags):

        vs = self.critic(states)
        vs_ = self.critic(next_states)

        deltas = rewards + self.gamma * vs_ * (1 - done_flags) - vs * (1 - done_flags)

        advantages = torch.zeros_like(deltas, dtype=torch.float)
        advantages[-1] = deltas[-1]
        for i in reversed(range(len(deltas) - 1)):
            advantages[i] = deltas[i] + self.gamma * self.gae_lambda * advantages[i + 1] * (1 - done_flags[i])

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        td_targets = advantages + vs

        return advantages, td_targets

    def train(self, max_steps):

        while self.total_steps < max_steps:

            with torch.no_grad():
                states, actions, rewards, next_states, old_action_probs, done_flags = self.rollout()
                advantages, targets = self.compute_advantages(states, next_states, rewards, done_flags)

            for _ in range(self.num_epochs):
                action_probs = self.actor.forward(states, softmax_dim=1)

                taken_action_probs = action_probs.gather(1, actions)

                importance_sampling_ratios = torch.exp(torch.log(taken_action_probs) - torch.log(old_action_probs))
                surrogate_loss_1 = importance_sampling_ratios * advantages
                surrogate_loss_2 = torch.clamp(importance_sampling_ratios, 1 - self.clip_rate,
                                               1 + self.clip_rate) * advantages
                surrogate_loss = -torch.min(surrogate_loss_1, surrogate_loss_2)

                actor_loss = surrogate_loss.mean()
                critic_loss = torch.nn.MSELoss()(self.critic(states), targets)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            if self.evaluate:
                avg_reward = self.test(3)
                print(f"timesteps: {self.total_steps}, score: {avg_reward:.2f}")

    def write_data(self, rewards, actor_loss, critic_loss):
        with pd.ExcelWriter("data.xlsx", mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            df = pd.DataFrame(
                [{'steps': self.total_steps, 'rewards': rewards, 'a_loss': actor_loss, 'c_loss': critic_loss}])
            df.to_excel(writer, sheet_name="Sheet1", header=False, startrow=writer.sheets["Sheet1"].max_row,
                        index=False)

    def save(self, steps):
        torch.save(self.critic.state_dict(), f"./model/ppo_critic{steps}.pth")
        torch.save(self.actor.state_dict(), f"./model/ppo_actor{steps}.pth")

    def load(self, steps):
        self.critic.load_state_dict(torch.load(f"./model/ppo_critic{steps}.pth"))
        self.actor.load_state_dict(torch.load(f"./model/ppo_actor{steps}.pth"))



