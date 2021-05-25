import gym
from collections import deque
import itertools
import numpy as np
from metrics import MetricLogger
import time
import dqn
import sarsa

GAMMA = 0.90
BATCH_SIZE = 500
BUFFER_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

env_name = "LunarLander-v2"
env = gym.make(env_name)

reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

# Main training loop
logger = MetricLogger()
episode = 0
agent = sarsa.Agent()
agent.agent_init(
    {
        "feature_shape": env.observation_space.shape,
        "action_space": env.action_space,
        "num_actions": env.action_space.n,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA,
        "update_target_freq": TARGET_UPDATE_FREQ,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
    },
    sarsa.Network(
        input_shape=env.observation_space.shape,
        output_dim=env.action_space.n,
    ),
)
state = env.reset()
action = agent.agent_start(state)


def show(count: int):
    new_env = gym.make(env_name)
    state = new_env.reset()
    while count > 0:
        action = agent.policy(state)
        state, _, terminal, _ = new_env.step(action)
        new_env.render()
        if terminal:
            count -= 1
            state = new_env.reset()
    new_env.close()


for step in itertools.count():
    new_state, reward, terminal, _ = env.step(action)
    episode_reward += reward
    if terminal:
        episode += 1
        if episode % 20 == 0:
            show(1)
        agent.agent_end(reward)
        reward_buffer.append(episode_reward)
        episode_reward = 0.0
        logger.log_episode()
        new_state = env.reset()
        action = agent.agent_start(new_state)
    else:
        action = agent.agent_step(reward, new_state, step)

    state = new_state
    logger.log_step(reward, agent.last_loss)
    # Logging
    if step % 1000 == 0:
        logger.record(episode=episode, epsilon=agent.last_epsilon, step=step)
        print()
