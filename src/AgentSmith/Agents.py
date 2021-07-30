import os
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from typing import Union, List, Literal, Tuple, Dict
from . import utils
from .utils import Transitions, soft_network_update, plot_returns


def set_seed(seed):

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    utils.set_seed(seed)


class DeepQ:
    def __init__(self,
                 env,
                 network: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=lambda x: x,
                 action_processor=lambda x: x,
                 greedy_function=lambda x: 0.2,
                 replay_size: int = 50000,
                 batch_size: int = 64,
                 target_update_rate: int = 1000,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 ) -> None:
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.greedy_function = greedy_function
        self.batch_size = batch_size
        self.target_update_rate = target_update_rate
        self.gamma = gamma
        self.tau = tau

        self.action_size = env.action_space.n
        self.replay_buffer = Transitions(
            env.observation_space.shape[0], replay_size, self.batch_size, self.device)

        self.target_network = copy.deepcopy(self.network)
        self.target_network.to(self.device)
        soft_network_update(self.network, self.target_network, 1)
        self.target_network.eval()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.uniform(0, 1) > epsilon:
            with torch.no_grad():
                self.network.eval()
                action = torch.argmax(self.network(
                    torch.from_numpy(state.astype(np.float32)))).item()
                return self.action_processor(action)
        else:
            return np.random.randint(self.action_size)

    def optimize(self) -> float:
        self.network.train()
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sampleTensors()

        q = torch.take_along_dim(self.network(
            batch_states), batch_actions.long().view(-1, 1), dim=1)

        expected_q = batch_rewards + self.gamma * \
            self.target_network(batch_next_states).amax(
                dim=1).unsqueeze(1) * (1 - batch_dones.float())

        loss = self.loss_function(q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.step_count % self.target_update_rate) == 0:
            soft_network_update(self.network, self.target_network, self.tau)

        return loss.item()

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, show_plot: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        if not os.path.exists('models'):
            os.makedirs('models')
        episode_returns = []
        step_losses = []
        episode_average_window = deque(maxlen=average_window)
        self.step_count = 0
        for self.episode in range(max_episodes):
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                self.step_count += 1

                if render_rate > 0 and (self.episode % render_rate) == 0:
                    self.env.render()

                action = self.select_action(
                    state, self.greedy_function(self.episode))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processor(next_state)
                self.replay_buffer.store(
                    state, action, reward, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    step_losses.append(self.optimize())

                state = next_state
                episode_return += reward

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (self.episode + 1, average_window, np.mean(episode_average_window)), end='')

            loss_dict = {'losses': step_losses, 'names': ['network']}
            plot_returns(name, episode_returns, show=show_plot,
                         average_window=average_window, loss_dict=loss_dict)

            if save_rate != 0 and (self.episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, self.episode + 1))

            if len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal:
                break
        print('\n')


class ActorCritic:
    def __init__(self,
                 env,
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 actor_optimizer: optim.Optimizer,
                 critic_optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=lambda x: x,
                 action_processor=lambda x: x,
                 gamma: float = 0.99,
                 entropy_function=lambda x: 0.01,
                 ) -> None:

        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.loss_function = loss_function
        self.device = device
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.gamma = gamma
        self.entropy_function = entropy_function

        self.transitions = []

    def select_action(self, state: np.ndarray) -> Tuple[float, float]:
        self.actor_network.eval()
        action = list(self.actor_network(
            torch.from_numpy(state.astype(np.float32))))
        action[0] = self.action_processor(action[0])
        return action

    def optimize(self) -> Tuple[float, float]:
        self.actor_network.train()
        self.critic_network.train()

        entropy_weight = self.entropy_function(self.episode)

        state, action, reward, next_state, done = self.transitions
        state = torch.from_numpy(state.astype(np.float32))
        next_state = torch.from_numpy(next_state.astype(np.float32))
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
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, show_plot: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        episode_returns = []
        step_losses = []
        episode_average_window = deque(maxlen=average_window)
        self.step_count = 0
        self.episode = -1
        while True:
            self.episode += 1
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                self.step_count += 1
                if render_rate > 0 and (self.episode % render_rate) == 0:
                    self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action[0])
                next_state = self.state_processor(next_state)
                self.transitions = [state, action, reward, next_state, done]

                losses = self.optimize()

                state = next_state
                episode_return += reward
                step_losses.append(losses)

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (self.episode + 1, average_window, np.mean(episode_average_window)), end='')

            loss_dict = {'losses': step_losses, 'names': ['actor', 'critic']}
            plot_returns(name, episode_returns, show=show_plot,
                         average_window=average_window, loss_dict=loss_dict)

            if save_rate != 0 and (self.episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, self.episode + 1))

            if (self.episode + 1) >= max_episodes or (len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal):
                break
        print('\n')


class DDPG:
    def __init__(self,
                 env,
                 actor_network: nn.Module,
                 critic_network: nn.Module,
                 actor_optimizer: optim.Optimizer,
                 critic_optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=lambda x: x,
                 action_processor=lambda x: x,
                 noise_function=lambda x: 0.025,
                 replay_size: int = 50000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 ) -> None:
        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.loss_function = loss_function
        self.device = device
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.noise_function = noise_function
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.action_size = env.action_space.shape[0]
        self.replay_buffer = Transitions(
            env.observation_space.shape[0], replay_size, self.batch_size, self.device)

        self.actor_target_network = copy.deepcopy(self.actor_network)
        self.critic_target_network = copy.deepcopy(self.critic_network)
        soft_network_update(self.actor_network, self.actor_target_network, 1)
        soft_network_update(self.critic_network, self.critic_target_network, 1)

    def select_action(self, state: np.ndarray, noise: float) -> float:
        self.actor_network.eval()
        action = self.actor_network(torch.from_numpy(
            state.astype(np.float32)).to(self.device))
        action = np.random.normal(action.item(), noise)

        return self.action_processor(action)

    def optimize(self) -> None:
        self.actor_network.train()
        self.critic_network.train()

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sampleTensors()

        values = self.critic_network(batch_states, batch_actions)
        next_actions = self.action_processor(
            self.actor_target_network(batch_next_states))
        target_values = batch_rewards + self.gamma * self.critic_target_network(
            batch_next_states, next_actions) * (1 - batch_dones)

        critic_loss = self.loss_function(values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = - \
            self.critic_network(
                batch_states, self.action_processor(self.actor_network(batch_states))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_network_update(self.actor_network,
                            self.actor_target_network, self.tau)
        soft_network_update(self.critic_network,
                            self.critic_target_network, self.tau)

        return actor_loss.item(), critic_loss.item()

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, show_plot: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        episode_returns = []
        step_losses = []
        episode_average_window = deque(maxlen=average_window)
        self.step_count = 0
        for self.episode in range(max_episodes):
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                self.step_count += 1
                if render_rate > 0 and (self.episode % render_rate) == 0:
                    self.env.render()
                action = self.select_action(
                    state, self.noise_function(self.episode))
                next_state, reward, done, _ = self.env.step([action])
                next_state = self.state_processor(next_state)
                self.replay_buffer.store(
                    state, action, reward, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    step_losses.append(self.optimize())

                state = next_state
                episode_return += reward

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (self.episode + 1, average_window, np.mean(episode_average_window)), end='')
            loss_dict = {'losses': step_losses, 'names': ['actor', 'critic']}
            plot_returns(name, episode_returns, show=show_plot,
                         average_window=average_window, loss_dict=loss_dict)

            if save_rate != 0 and (self.episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, self.episode + 1))

            if len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal:
                break
        print('\n')


class TD3:
    def __init__(self,
                 env,
                 actor_network: nn.Module,
                 critic_network1: nn.Module,
                 critic_network2: nn.Module,
                 actor_optimizer: optim.Optimizer,
                 critic_optimizer: optim.Optimizer,
                 loss_function,
                 device: Literal['cpu', 'gpu'] = 'cpu',
                 state_processor=lambda x: x,
                 action_processor=lambda x: x,
                 target_noise_function=lambda x: 0.025,
                 target_noise_clip: float = 0.5,
                 noise_function=lambda x: 0.025,
                 replay_size: int = 50000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 update_delay: int = 2,
                 ) -> None:

        self.env = env
        self.actor_network = actor_network
        self.critic_network1 = critic_network1
        self.critic_network2 = critic_network2
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.loss_function = loss_function
        self.device = device
        self.state_processor = state_processor
        self.action_processor = action_processor
        self.target_noise_function = target_noise_function
        self.target_noise_clip = target_noise_clip
        self.noise_function = noise_function
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_delay = update_delay

        self.action_size = env.action_space.shape[0]
        self.replay_buffer = Transitions(
            env.observation_space.shape[0], replay_size, self.batch_size, self.device)

        self.actor_target_network = copy.deepcopy(self.actor_network)
        self.critic_target_network1 = copy.deepcopy(self.critic_network1)
        self.critic_target_network2 = copy.deepcopy(self.critic_network2)
        soft_network_update(self.actor_network, self.actor_target_network, 1)
        soft_network_update(self.critic_network1,
                            self.critic_target_network1, 1)
        soft_network_update(self.critic_network2,
                            self.critic_target_network2, 1)

    def select_action(self, state: np.ndarray) -> float:
        self.actor_network.eval()

        action = self.actor_network(torch.from_numpy(
            state.astype(np.float32)).to(self.device))

        noise = self.noise_function()
        action = action.item() + noise

        return self.action_processor(action)

    def optimize(self) -> None:

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sampleTensors()

        noise = torch.FloatTensor(
            self.target_noise_function()).to(self.device)
        clipped_noise = torch.clamp(
            noise, -self.target_noise_clip, self.target_noise_clip)

        values1 = self.critic_network1(batch_states, batch_actions)
        values2 = self.critic_network2(batch_states, batch_actions)
        next_actions = self.action_processor(self.actor_target_network(
            batch_next_states) + clipped_noise)
        q1 = self.critic_target_network1(batch_next_states, next_actions)
        q2 = self.critic_target_network2(batch_next_states, next_actions)
        q = torch.min(q1, q2)

        target_value = batch_rewards + self.gamma * q * (1 - batch_dones)

        critic1_loss = self.loss_function(values1, target_value.detach())
        critic2_loss = self.loss_function(values2, target_value.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if (self.step_count % self.update_delay):
            actor_loss = - \
                self.critic_network1(
                    batch_states, self.action_processor(self.actor_network(batch_states))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_network_update(self.critic_network1,
                                self.critic_target_network1, self.tau)
            soft_network_update(self.critic_network2,
                                self.critic_target_network2, self.tau)
            soft_network_update(self.actor_network,
                                self.actor_target_network, self.tau)
            return actor_loss.item(), critic_loss.item()
        return 0, critic_loss.item()

    def train(self, name: str = 'agent', average_window: int = 20, max_episodes: int = 100, return_goal: float = 10e9, show_plot: bool = False, render_rate: int = 0, save_rate: int = 10) -> None:
        episode_returns = []
        step_losses = []
        episode_average_window = deque(maxlen=average_window)
        self.step_count = 0
        for self.episode in range(max_episodes):
            done = False
            episode_return = 0
            state = self.state_processor(self.env.reset())
            while done is not True:
                self.step_count += 1
                if render_rate > 0 and (self.episode % render_rate) == 0:
                    self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processor(next_state)
                self.replay_buffer.store(
                    state, action, reward, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    step_losses.append(self.optimize())

                state = next_state
                episode_return += reward

            episode_returns.append(episode_return)
            episode_average_window.append(episode_return)
            print('\rEpisode: %8u, Return (%8u episode averaged): %8u' %
                  (self.episode + 1, average_window, np.mean(episode_average_window)), end='')

            loss_dict = {'losses': step_losses, 'names': ['actor', 'critic']}
            plot_returns(name, episode_returns, show=show_plot,
                         average_window=average_window, loss_dict=loss_dict)

            if save_rate != 0 and (self.episode % save_rate) == 0:
                torch.save(self.target_network.state_dict(), 'models/%s_%08u.pt' %
                           (name, self.episode + 1))

            if len(episode_returns) > average_window and np.mean(episode_returns[:-average_window]) >= return_goal:
                break
        print('\n')
