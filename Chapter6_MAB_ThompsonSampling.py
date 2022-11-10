import gym
import gym_bandits
import numpy as np

env = gym.make("BanditTwoArmedHighLowFixed-v0")

# Environment
# print(env.action_space.n) : 2 Arms (0, 1)
# print(env.p_dist)
# Arm 1 : 80% (Win) / 20% (Lose)
# Arm 2 : 20% (Win) / 80% (Lose)

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
alpha = np.ones(2)
beta = np.ones(2)
num_rounds = 100

def thompson_sampling(alpha, beta):
    samples = [np.random.beta(alpha[i] + 1, beta[i] + 1)
               for i in range(2)]
    return np.argmax(samples)

for i in range(num_rounds):
    arm = thompson_sampling(alpha, beta)
    next_state, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]
    if reward == 1:
        alpha[arm] += 1
    else:
        beta[arm] += 1
    print(Q, alpha, beta)

print()
print("[Arm 1 Average Reward, Arm 2 Average Reward]")
print(Q)
print()
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))