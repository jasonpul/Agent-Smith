import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from typing import Union, List, Literal, Tuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class Transitions:
    def __init__(self, maxlen: int) -> None:
        self.transitions = deque([], maxlen=maxlen)

    def append(self, transition) -> None:
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index) -> Transition:
        return self.transitions[index]

    def sample(self, count: int) -> List[Transition]:
        return random.sample(self.transitions, count)

    def sampleTensors(self, count: int, device: Literal['cpu', 'gpu']) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transitions = random.sample(self.transitions, count)

        states = torch.cat(
            [i.state for i in transitions])
        actions = torch.tensor(
            [i.action for i in transitions], device=device)
        rewards = torch.tensor(
            [i.reward for i in transitions], device=device)
        next_states = torch.cat(
            [i.next_state for i in transitions])
        dones = torch.tensor(np.array(
            [i.done for i in transitions], dtype=int), device=device)

        states = states if len(states.shape) > 1 else states.view(count, -1)
        actions = actions if len(
            actions.shape) > 1 else actions.view(count, -1)
        rewards = rewards if len(
            rewards.shape) > 1 else rewards.view(count, -1)
        next_states = next_states if len(
            next_states.shape) > 1 else next_states.view(count, -1)
        dones = dones if len(dones.shape) > 1 else dones.view(count, -1)

        return states, actions, rewards, next_states, dones


class DeepQAgent:
    def __init__(self,
                 env,
                 policy_network: nn.Module,
                 target_network: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=None,
                 greedy_function=lambda x: 0.2,
                 replay_size: int = 50000,
                 batch_size: int = 64,
                 gamma: float = 0.95,
                 target_update_rate: int = 1000) -> None:
        self.env = env
        self.policy_network = policy_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.greedy_function = greedy_function
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.action_size = env.action_space.n

        self.replay_buffer = Transitions(replay_size)

        if state_processor is not None:
            self.state_processor = state_processor
        else:
            self.state_processor = lambda x: torch.tensor(
                x, dtype=torch.float32, device=device)

        self.update_target_network()
        self.target_network.eval()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        if random.uniform(0, 1) > epsilon:
            with torch.no_grad():
                self.policy_network.eval()
                action = torch.argmax(self.policy_network(state)).item()
                return action
        else:
            return np.random.randint(self.action_size)

    def optimize(self) -> None:
        self.policy_network.train()
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sampleTensors(
            self.batch_size, self.device)

        q = torch.take_along_dim(self.policy_network(
            batch_states), batch_actions.view(-1, 1), dim=1)

        expected_q = batch_rewards + self.gamma * \
            self.target_network(batch_next_states).amax(
                dim=1).unsqueeze(1) * (1 - batch_dones.float())

        loss = self.loss_function(q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_returns(self, name: str, returns: List[int], average_window: int, show: bool = True) -> None:
        plt.clf()
        plt.plot(returns, label='Episode Returns')

        if len(returns) >= average_window:
            y = np.convolve(returns, np.ones(average_window),
                            'valid') / average_window
            x = np.arange(y.shape[0]) + average_window
            plt.plot(x, y, label='%u Episode Avg. Returns' % average_window)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend(loc='upper left')
        plt.savefig('%s_returns.png' % name)

        if show:
            plt.ion()
            plt.figure(1)
            plt.show()
            plt.pause(0.001)

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, plot_returns: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        if not os.path.exists('models'):
            os.makedirs('models')
        episode_returns = []
        episode_average_window = deque(maxlen=average_window)
        step_count = 0
        for episode in range(max_episodes):
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                step_count += 1

                if render_rate > 0 and (episode % render_rate) == 0:
                    self.env.render()

                action = self.select_action(
                    state, self.greedy_function(episode))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processor(next_state)
                self.replay_buffer.append(Transition(
                    state, action, reward, next_state, done))

                if len(self.replay_buffer) >= self.batch_size:
                    self.optimize()

                if (step_count % self.target_update_rate) == 0:
                    self.update_target_network()

                state = next_state
                episode_return += reward

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (episode + 1, average_window, np.mean(episode_average_window)), end='')

            self.plot_returns(name, episode_returns,
                              average_window, plot_returns)

            if save_rate != 0 and (episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, episode + 1))

            if len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal:
                break
        print('\n')


class ActorCriticAgent:

    def __init__(self,
                 env,
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 actor_optimizer: optim.Optimizer,
                 critic_optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=None,
                 gamma: float = 0.95,
                 entropy_function=lambda x: 0.01,
                 ) -> None:

        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.loss_function = loss_function
        self.device = device
        self.gamma = gamma
        self.entropy_function = entropy_function

        self.transitions = Transitions(1)

        if state_processor is not None:
            self.state_processor = state_processor
        else:
            self.state_processor = lambda x: torch.tensor(
                x, dtype=torch.float32, device=device)

    def select_action(self, state: torch.Tensor) -> Tuple[float, float]:
        self.actor_network.eval()

        return self.actor_network(state)

    def optimize(self, entropy_weight: float) -> Tuple[float, float]:
        self.actor_network.train()
        self.critic_network.train()

        transition = self.transitions[0]
        state, action, reward, next_state, done = transition.state, transition.action, transition.reward, transition.next_state, transition.done
        log_probability = action[1].log_prob(action[0]).sum(dim=-1)

        predicted_value = self.critic_network(state)
        target_value = reward + self.gamma * \
            self.critic_network(next_state) * (1 - done)

        critic_loss = self.loss_function(
            predicted_value, target_value.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        advantage = (target_value - predicted_value).detach()
        actor_loss = - advantage * log_probability
        actor_loss += -entropy_weight * log_probability
        self.actor_network.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def plot_returns(self, name: str, returns: List[int], average_window: int, losses: List, show: bool = True) -> None:
        losses = np.array(losses)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(returns, label='Episode Returns')
        plt.subplot(3, 1, 2)
        plt.plot(losses[:, 0], label='Actor Losses')
        plt.subplot(3, 1, 3)
        plt.plot(losses[:, 1], label='Critic Losses')

        if len(returns) >= average_window:
            y = np.convolve(returns, np.ones(average_window),
                            'valid') / average_window
            x = np.arange(y.shape[0]) + average_window
            plt.subplot(3, 1, 1)
            plt.plot(x, y, label='%u Episode Avg. Returns' % average_window)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend(loc='upper left')
        plt.savefig('%s_returns.png' % name)

        if show:
            plt.ion()
            plt.figure(1)
            plt.show()
            plt.pause(0.001)

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, plot_returns: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        episode_returns = []
        step_losses = []
        episode_average_window = deque(maxlen=average_window)
        step_count = 0
        episode = -1
        while True:
            episode += 1
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                step_count += 1
                if render_rate > 0 and (episode % render_rate) == 0:
                    self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(
                    action[0].cpu().detach().numpy())
                next_state = self.state_processor(next_state)
                self.transitions.append(Transition(
                    state, action, reward, next_state, done))

                losses = self.optimize(self.entropy_function(episode))

                state = next_state
                episode_return += reward
                step_losses.append(losses)

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (episode + 1, average_window, np.mean(episode_average_window)), end='')

            self.plot_returns(name, episode_returns,
                              average_window, step_losses, plot_returns)

            if save_rate != 0 and (episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, episode + 1))

            if (episode + 1) >= max_episodes or (len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal):
                break
        print('\n')
