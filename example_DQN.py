from src.AgentSmith import Agents
from src.AgentSmith import utils
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List

device = 'cpu'


class DenseNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int]):
        super(DenseNetwork, self).__init__()

        self.input_size, self.output_size = input_size, output_size
        self.hidden_layers = hidden_layers

        sequential = [nn.Linear(self.input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers) - 1):
            in_features = self.hidden_layers[i]
            out_features = self.hidden_layers[i + 1]
            sequential += [nn.Linear(in_features, out_features), nn.ReLU()]
        sequential.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        self.stack = nn.Sequential(*tuple(sequential))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        return self.stack(x)


env = gym.make('CartPole-v0')

seed = 42
torch.manual_seed(seed)
env.seed(seed)
Agents.set_seed(seed)

network = DenseNetwork(4, 2, [32, 24]).to(device)
optimizer = optim.Adam(network.parameters(), lr=0.001)
loss_function = F.mse_loss
greedy_function = utils.log_decay_function(60, 5)

agent = Agents.DeepQ(
    env=env,
    network=network,
    optimizer=optimizer,
    loss_function=loss_function,
    device=device,
    greedy_function=greedy_function,
    target_update_rate=10,
    tau=1.0
)
agent.train(show_plot=False, max_episodes=200, render_rate=0, save_rate=0)
