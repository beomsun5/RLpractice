import gym
import pandas as pd

env = gym.make('FrozenLake-v0')

def random_policy():
    return env.action_space.sample()

V = {}
for s in range(env.observation_space.n):
    V[s] = 0.0

alpha = 0.85 # Learning Rate
gamma = 0.90 # Discount Factor

num_episodes = 50000
num_timesteps = 1000

# Compute the values of the states
for i in range(num_episodes):
    s = env.reset()
    for t in range(num_timesteps):
        a = random_policy()
        s_, r, done, _ = env.step(a)
        V[s] += alpha * (r + gamma * V[s_] - V[s])
        s = s_  # Update the state
        if done:
            break

df = pd.DataFrame(list(V.items()), columns=['state', 'value'])
print(df)