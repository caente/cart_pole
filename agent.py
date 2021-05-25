from torch import nn
import torch
from collections import deque
import numpy as np
import random
import copy


class Agent:
    def __init__(self, num_actions, feature_shape, buffer_size, batch_size, network) -> None:
        self.use_cuda = torch.cuda.is_available()
        self.online_net = network
        self.target_net = copy.deepcopy(network)
        self.target_net.load_state_dict(self.online_net.state_dict())
        if self.use_cuda:
            self.online_net = self.online_net.to(device="cuda")
            self.target_net = self.target_net.to(device="cuda")
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)
        self.terminal_state = np.zeros(feature_shape)
        self.num_actions = num_actions
        self.last_loss = 0.0
        self.rand_generator = np.random.RandomState()
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size


    def agent_init(self, agent_config):
        pass

    def policy(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if self.use_cuda:
            state_t = state_t.cuda()
        return self.choose_action(state_t)

    def choose_action(self, state_t):
        pass

    def agent_start(self, state):
        action = self.policy(state)
        self.last_action = action
        self.last_state = state
        return action

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

    def td_target(self, rewards, terminals, new_states):
        pass

    def optimize(self, estimate, targets):
        loss = nn.functional.smooth_l1_loss(estimate, targets)
        self.last_loss = loss.cpu().detach().numpy()
        # Gradient descend
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        states, actions, rewards, terminals, new_states = self.sample_transitions()

        # Compute target
        targets = self.td_target(rewards, terminals, new_states)

        # Compute estimate
        q_values = self.online_net(states)
        estimated_q_values = torch.gather(input=q_values, dim=1, index=actions)
        self.optimize(estimated_q_values, targets)

    def sync_networks(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def agent_step(self, reward, state, step):
        action = self.policy(state)
        transition = (self.last_state, self.last_action, reward, 0, state)
        self.replay_buffer.append(transition)
        self.learn()
        self.last_action = action
        self.last_state = state
        self.sync_networks()
        return action

    def agent_end(self, reward):
        transition = (self.last_state, self.last_action, reward, 1, self.terminal_state)
        self.replay_buffer.append(transition)
        self.learn()
