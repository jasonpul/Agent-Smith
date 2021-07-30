import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.AgentSmith import Agents


device = 'cpu'


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class Critic(nn.Module):
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


class GaussianNoise:
    """Gaussian Noise.
    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialize."""
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)


env = gym.make('Pendulum-v0')
env = ActionNormalizer(env)
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

seed = 777
torch.manual_seed(seed)
env.seed(seed)
Agents.set_seed(seed)

actor_network = Actor(state_size, action_size).to(device)
critic_network1 = Critic(state_size + action_size).to(device)
critic_network2 = Critic(state_size + action_size).to(device)

actor_optimizer = optim.Adam(actor_network.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(
    list(critic_network1.parameters()) + list(critic_network2.parameters()), lr=1e-3)

loss_function = F.mse_loss

target_noise_function = GaussianNoise(action_size, 0.2, 0.2).sample
noise_function = GaussianNoise(action_size, 0.1, 0.1).sample


def action_processor(x):
    if type(x) == torch.Tensor:
        return x.clamp(-1, 1)
    else:
        return np.clip(x, -1, 1)


agent = Agents.TD3(
    env=env,
    actor_network=actor_network,
    critic_network1=critic_network1,
    critic_network2=critic_network2,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    loss_function=loss_function,
    device=device,
    action_processor=action_processor,
    target_noise_function=target_noise_function,
    target_noise_clip=0.5,
    noise_function=noise_function,
    gamma=0.99,
    tau=0.005,
    replay_size=100000,
    batch_size=128
)
agent.train(show_plot=False, max_episodes=100,
            render_rate=0, save_rate=0)
