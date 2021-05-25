from torch import nn
import torch
from collections import deque
import numpy as np
import random
from torch.tensor import Tensor


class Network(nn.Module):
    def __init__(self, network_config):
        super().__init__()
        in_features = int(np.prod(network_config["input_dim"]))
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, network_config["output_dim"]),
        )

    def forward(self, x: Tensor):
        return self.net(x)

class Agent:
    def __init__(self) -> None:
        pass

    def agent_init(self, agent_config):
        """
        agent_config:
        {
            network_config: {
                input_dim: shape,
                output_dim: int
            }
            buffer_size: integer,
            batch_size: integer,
            action_space: ndarray,
            epsilon_start: float,
            epsilon_end: float,
            epsilon_decay: float
            gamma: float,
            update_target_freq: float
        }
        """

        self.use_cuda = torch.cuda.is_available()
        self.action_space = agent_config["action_space"]
        self.epsilon_start = agent_config["epsilon_start"]
        self.epsilon_end = agent_config["epsilon_end"]
        self.epsilon_decay = agent_config["epsilon_decay"]
        self.gamma = agent_config["gamma"]
        self.update_target_freq = agent_config["update_target_freq"]

        self.online_net = Network(agent_config["network_config"])
        self.target_net = Network(agent_config["network_config"])
        self.target_net.load_state_dict(self.online_net.state_dict())

        if self.use_cuda:
            self.online_net = self.online_net.to(device="cuda")
            self.target_net = self.target_net.to(device="cuda")

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

        self.replay_buffer = deque(maxlen=agent_config["buffer_size"])
        self.batch_size = agent_config["batch_size"]
        self.last_loss = 0.0
        self.terminal_state = np.zeros(agent_config["network_config"]["input_dim"])

    def policy(self, state, epsilon=-1):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = self.action_space.sample()
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

    def learn(self):
        states, actions, rewards, terminals, new_states = self.sample_transitions()

        # Compute target
        target_q_values = self.target_net(new_states)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards + self.gamma * (1 - terminals) * max_target_q_values

        # Compute loss
        q_values = self.online_net(states)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        self.last_loss = loss.cpu().detach().numpy()
        # Gradient descend
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def agent_step(self, reward, state, step):
        epsilon = self.get_epsilon(step)
        action = self.policy(state, epsilon)
        transition = (self.last_state, self.last_action, reward, 0, state)
        self.replay_buffer.append(transition)
        self.learn()
        self.last_action = action
        self.last_state = state
        self.last_epsilon = epsilon
        self.target_net.load_state_dict(self.online_net.state_dict())
        return action

    def agent_end(self, reward):
        transition = (self.last_state, self.last_action, reward, 1, self.terminal_state)
        self.replay_buffer.append(transition)
        self.learn()
