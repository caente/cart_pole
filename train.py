import gym
from collections import deque
import itertools
import numpy as np
from metrics import MetricLogger
import dqn

GAMMA = 0.90
BATCH_SIZE = 100
BUFFER_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

env = gym.make("CartPole-v1")

reward_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

# Main training loop
logger = MetricLogger()
episode = 0
agent = dqn.Agent()
agent.agent_init(
    {
        "action_space": env.action_space,
        "network_config": {
            "input_dim": env.observation_space.shape,
            "output_dim": env.action_space.n,
        },
        "action_space": env.action_space,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay": EPSILON_DECAY,
        "gamma": GAMMA,
        "update_target_freq": TARGET_UPDATE_FREQ,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
    }
)
state = env.reset()
action = agent.agent_start(state)
for step in itertools.count():
    new_state, reward, terminal, _ = env.step(action)
    episode_reward += reward
    if terminal:
        episode += 1
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

    # After solved, watch it
    if len(reward_buffer) >= 100:
        if np.mean(reward_buffer) >= 195:
            break
            count = 0
            while True:
                count += 1
                action = agent.policy(state)
                state, _, terminal, _ = env.step(action)
                env.render()
                if terminal:
                    print(f"FAILED! After {count} steps")
                    count = 0
                    state = env.reset()
