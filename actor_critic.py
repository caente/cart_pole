from torch import nn
import torch
from collections import deque
import numpy as np
import random
from torch.tensor import Tensor
import copy
import agent


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        in_features = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=0),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class StateValueNetwork(nn.Module):
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


class Agent:
    def __init__(
        self,
        num_actions,
        feature_shape,
        buffer_size,
        batch_size,
        network_policy,
        network_state_values,
    ) -> None:
        self.use_cuda = torch.cuda.is_available()

        self.online_policy = network_policy
        self.target_policy = copy.deepcopy(self.online_policy)
        self.target_policy.load_state_dict(self.online_policy.state_dict())
        self.online_state_values = network_state_values
        self.target_state_values = copy.deepcopy(self.online_state_values)
        self.target_state_values.load_state_dict(self.online_state_values.state_dict())

        if self.use_cuda:
            self.online_policy = self.online_policy.to(device="cuda")
            self.target_policy = self.target_policy.to(device="cuda")
            self.online_state_values = self.online_state_values.to(device="cuda")
            self.target_state_values = self.target_state_values.to(device="cuda")

        self.policy_optimizer = torch.optim.Adam(
            self.online_policy.parameters(), lr=5e-3
        )
        self.values_optimizer = torch.optim.Adam(
            self.online_state_values.parameters(), lr=5e-2
        )
        self.terminal_state = np.zeros(feature_shape)
        self.num_actions = num_actions
        self.last_loss = 0.0
        self.rand_generator = np.random.RandomState()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.I = 1.0

    def agent_init(self, agent_config):
        """
        agent_config:
        {
            gamma: float
        }
        """
        self.gamma = agent_config["gamma"]

    def policy(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if self.use_cuda:
            state_t = state_t.cuda()
        return self.choose_action(state_t)

    def choose_action(self, state_t):
        out = self.online_policy(state_t)
        probs = torch.exp(out).squeeze().cpu().detach().numpy()
        action = self.rand_generator.choice(self.num_actions, p=probs)
        return action

    def td_target(self, rewards, terminals, new_states):
        target_values = self.target_state_values(new_states)
        target_values = rewards + self.gamma * (1 - terminals) * target_values
        return target_values

    def agent_start(self, state):
        action = self.policy(state)
        self.last_action = action
        self.last_state = state
        self.I = 1.0
        return action

    def learn(self):
        states, actions, rewards, terminals, new_states = self.sample_transitions()

        # Compute target
        target_values = self.td_target(rewards, terminals, new_states)
        # Compute estimate
        state_values = self.online_state_values(states)
        td_error = nn.functional.smooth_l1_loss(state_values, target_values)
        self.values_optimizer.zero_grad()
        td_error.backward()
        self.values_optimizer.step()

        estimated_action_probs = self.online_policy(states)
        estimated_action_weigth = torch.gather(
            input=estimated_action_probs, dim=1, index=actions
        )
        nll_loss = nn.NLLLoss()
        policy_error = nll_loss(
            estimated_action_probs * td_error.detach() * self.I, actions.squeeze()
        )

        self.last_loss = policy_error.cpu().detach().numpy()
        self.policy_optimizer.zero_grad()
        policy_error.backward()
        self.policy_optimizer.step()
        self.I *= self.gamma

    def agent_step(self, reward, state, step):
        action = self.policy(state)
        transition = (self.last_state, self.last_action, reward, 0, state)
        self.replay_buffer.append(transition)
        if step > 10:
            self.learn()
        self.last_action = action
        self.last_state = state
        self.sync_networks()
        return action

    def agent_end(self, reward):
        transition = (self.last_state, self.last_action, reward, 1, self.terminal_state)
        self.replay_buffer.append(transition)
        self.learn()

    def sync_networks(self):
        self.target_policy.load_state_dict(self.online_policy.state_dict())
        self.target_state_values.load_state_dict(self.online_state_values.state_dict())

    def sample_transitions(self):
        transitions = random.sample(
            self.replay_buffer, min(len(self.replay_buffer), self.batch_size)
        )
        states = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rewards = np.asarray([t[2] for t in transitions])
        terminals = np.asarray([t[3] for t in transitions])
        new_states = np.asarray([t[4] for t in transitions])

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        terminals_t = torch.as_tensor(terminals, dtype=torch.float32).unsqueeze(-1)
        new_states_t = torch.as_tensor(new_states, dtype=torch.float32)
        if self.use_cuda:
            states_t = states_t.cuda()
            actions_t = actions_t.cuda()
            rewards_t = rewards_t.cuda()
            terminals_t = terminals_t.cuda()
            new_states_t = new_states_t.cuda()
        return states_t, actions_t, rewards_t, terminals_t, new_states_t


def update_parameters(net, update):
    state_dict = net.state_dict()

    for name, param in state_dict.items():
        # Don't update if this is not a weight.
        if not "weight" in name:
            continue

        # Transform the parameter as required.
        transformed_param = param + update

        # Update the parameter.
        state_dict[name].copy_(transformed_param)
