from torch import nn
import torch
from collections import deque
import numpy as np
import random
from torch.tensor import Tensor
import copy
import agent


class Network(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        in_features = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Agent(agent.Agent):
    def agent_init(self, agent_config):
        """
        agent_config:
        {
            gamma: float
        }
        """
        self.gamma = agent_config["gamma"]

    def choose_action(self, state_t):
        q_values = self.online_net(state_t)
        probs = torch.softmax(q_values, dim=-1).squeeze().cpu().detach().numpy()
        action = self.rand_generator.choice(self.num_actions, p=probs)
        return action

    def td_target(self, rewards, terminals, new_states):
        target_q_values = self.target_net(new_states)
        probs = torch.softmax(target_q_values, dim=1)
        weighted_action_values = torch.sum(target_q_values * probs, axis=1).unsqueeze(
            -1
        )
        targets = rewards + self.gamma * (1 - terminals) * weighted_action_values
        return targets
