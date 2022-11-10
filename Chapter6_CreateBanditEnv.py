import gym
import gym_bandits

env = gym.make("BanditTwoArmedHighLowFixed-v0")
# Environment
# print(env.action_space.n) : 2 Arms (0, 1)
# print(env.p_dist)
# Arm 1 : 80% (Win) / 20% (Lose)
# Arm 2 : 20% (Win) / 80% (Lose)