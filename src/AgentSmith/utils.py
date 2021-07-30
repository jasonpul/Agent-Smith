import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Union, List, Literal, Tuple, Dict


def set_seed(seed):

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_network_update(network: nn.Module, target_network: nn.Module, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data)


def plot_returns(name: str, returns: List[int], average_window: int, loss_dict: Union[Dict, None] = None, show: bool = False) -> None:
    plt.clf()
    if loss_dict is not None and len(loss_dict['losses']) > 0:
        loss_names = loss_dict['names']
        losses = np.array(loss_dict['losses'])
        losses = losses if len(losses.shape) > 1 else np.expand_dims(losses, 1)
        main_subplot = (1 + len(loss_names), 1, 1)

        plt.subplot(main_subplot[0], 1, 1)
        plt.plot(returns, label='Episode Returns')

        for i in range(len(loss_names)):
            plt.subplot(main_subplot[0], 1, i + 2)
            plt.plot(losses[:, i], label=loss_names[i])

    else:
        main_subplot = (1, 1, 1)
        plt.subplot(1, 1, 1)
        plt.plot(returns, label='Episode Returns')

    if len(returns) >= average_window:
        y = np.convolve(returns, np.ones(average_window),
                        'valid') / average_window
        x = np.arange(y.shape[0]) + average_window
        plt.subplot(*main_subplot)
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


def log_decay_function(m: float, sd: float):
    return lambda x: 1 / (1 + (x / m) ** sd)


class Transitions:
    def __init__(self, state_dim: int, size: int, batch_size: int = 32, device: Literal['cpu', 'gpu'] = 'cpu'):
        self.states = np.zeros([size, state_dim], dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=int)
        self.max_size, self.batch_size, self.device = size, batch_size, device
        self.index, self.size, = 0, 0

    def store(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.states[self.index] = state
        self.next_states[self.index] = next_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(states=self.states[idxs],
                    next_states=self.next_states[idxs],
                    actions=self.actions[idxs],
                    rewards=self.rewards[idxs],
                    dones=self.dones[idxs])

    def sampleTensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.sample()
        states = torch.from_numpy(sample['states']).to(self.device)
        actions = torch.from_numpy(sample['actions']).to(self.device)
        rewards = torch.from_numpy(sample['rewards']).to(self.device)
        next_states = torch.from_numpy(sample['next_states']).to(self.device)
        dones = torch.from_numpy(sample['dones']).to(self.device)

        states = states if len(states.shape) > 1 else states.view(
            self.batch_size, -1)
        actions = actions if len(
            actions.shape) > 1 else actions.view(self.batch_size, -1)
        rewards = rewards if len(
            rewards.shape) > 1 else rewards.view(self.batch_size, -1)
        next_states = next_states if len(
            next_states.shape) > 1 else next_states.view(self.batch_size, -1)
        dones = dones if len(dones.shape) > 1 else dones.view(
            self.batch_size, -1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_state = self.next_states[index]
        done = self.dones[index]
        return state, action, reward, next_state, done
