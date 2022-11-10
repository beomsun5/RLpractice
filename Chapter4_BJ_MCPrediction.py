import gym
import pandas as pd
from collections import defaultdict

env = gym.make('Blackjack-v0')
# ------------------------------
#       MC Prediction Code
# ------------------------------

# Defining a policy
def policy(state):
    return 0 if state[0] > 19 else 1

# Generating the episode
num_timesteps = 100

def generate_episode(policy):
    episode = []
    state = env.reset()
    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode

# Computing the value function (MC Prediction Algorithm Code)
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 500000

for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        # First-Visit MC Prediction
        if state not in states[0:t]:
            R = (sum(rewards[t:]))
            total_return[state] = total_return[state] + R
            N[state] = N[state] + 1
        '''
        # Every-Visit MC Prediction
        R = (sum(rewards[t:]))
        total_return[state] = total_return[state] + R
        N[state] = N[state] + 1
        '''

# Show the result
total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state', 'N'])
df = pd.merge(total_return, N, on="state")
print(df.head(10))
print()
# Compute the value of the state as the average return
df['value'] = df['total_return']/df['N']
print(df.head(10))

# Example 1 (Check the estimated value of the state)
print()
print("Player Cards : 21 / Dealer Cards : 9 / Usable Ace : No")
print(df[df['state']==(21, 9, False)]['value'].values)

# Example 2
print()
print("Player Cards : 5 / Dealer Cards : 8 / Usable Ace : No")
print(df[df['state']==(5, 8, False)]['value'].values)
