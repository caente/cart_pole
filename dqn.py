from torch import nn
import torch
import numpy as np
import random
from torch.tensor import Tensor
import agent


class Network(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        in_features = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Agent(agent.Agent):
    def agent_init(self, agent_config):
        """
        agent_config:
        {
            action_space: ndarray,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay: float
            gamma: float
        }
        """

        self.epsilon_start = agent_config["epsilon_start"]
        self.epsilon_end = agent_config["epsilon_end"]
        self.epsilon_decay = agent_config["epsilon_decay"]
        self.gamma = agent_config["gamma"]

    def policy(self, state, epsilon=-1):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = self.rand_generator.choice(self.num_actions)
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32)
            if self.use_cuda:
                state_t = state_t.cuda()
            q_values = self.online_net(state_t.unsqueeze(0))
            max_q_index = torch.argmax(q_values, dim=1)[0]
            action = max_q_index.detach().item()
        return action

    def agent_start(self, state):
        action = self.policy(state, self.epsilon_start)
        self.last_action = action
        self.last_state = state
        self.last_epsilon = self.epsilon_start
        return action

    def get_epsilon(self, step):
        return np.interp(
            step, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end]
        )

    def td_target(self, rewards, terminals, new_states):
        target_q_values = self.target_net(new_states)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards + self.gamma * (1 - terminals) * max_target_q_values
        return targets

    def agent_step(self, reward, state, step):
        epsilon = self.get_epsilon(step)
        action = self.policy(state, epsilon)
        transition = (self.last_state, self.last_action, reward, 0, state)
        self.replay_buffer.append(transition)
        self.learn()
        self.last_action = action
        self.last_state = state
        self.last_epsilon = epsilon
        self.sync_networks()
        return action
