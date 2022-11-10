import gym
import numpy as np
import random
import pandas as pd

env = gym.make('FrozenLake-v0')

Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0

def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x : Q[(state, x)])

alpha = 0.85    # Learning Rate
gamma = 0.90    # Discount Factor
epsilon = 0.80  # Epsilon Value

num_episodes = 50000
num_timesteps = 1000

# Compute the policy
for i in range(num_episodes):
    s = env.reset()
    for t in range(num_timesteps):
        a = epsilon_greedy(s, epsilon)
        s_, r, done, _ = env.step(a)
        # Compute Q
        a_ = np.argmax(Q[(s_, a)] for a in range(env.action_space.n))
        Q[(s, a)] += alpha * (r + gamma * Q[(s_, a_)] - Q[(s, a)])
        s = s_
        a = a_
        if done:
            break

df = pd.DataFrame(list(Q.items()), columns=['state_action_pair', 'Q_value'])
print(df)