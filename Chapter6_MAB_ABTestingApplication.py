import matplotlib_inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

df = pd.DataFrame()
for i in range(5):
    df['Banner_type_' + str(i)] = np.random.randint(0, 2, 100000)

# print(df.head())

num_iterations = 100000
num_banners = 5
count = np.zeros(num_banners)
sum_rewards = np.zeros(num_banners)
Q = np.zeros(num_banners)
banner_selected = []

def epsilon_greedy_method(epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(num_banners)
    else:
        return np.argmax(Q)

# Run the bandit test
for i in range(num_iterations):
    banner = epsilon_greedy_method(0.5)
    reward = df.values[i, banner]
    count[banner] += 1
    sum_rewards[banner] += reward
    Q[banner] = sum_rewards[banner] / count[banner]
    banner_selected.append(banner)

print("The best banner is banner {}".format(np.argmax(Q)))

ax = sns.countplot(banner_selected)
ax.set(xlabel = 'Banner', ylabel = 'Count')
plt.show()