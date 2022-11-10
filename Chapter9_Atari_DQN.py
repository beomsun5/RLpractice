import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,\
    Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

env = gym.make("MsPacman-v0")
state_size = (88, 80, 1)
action_size = env.action_space.n

# Preprocess the game screen
# The raw game screen size is quite large, so computationally expensive.
# To avoid this, preprocess the game screen and feed it to the DQN
color = np.array([210, 164, 74]).mean()

def preprocess_state(state):
    image = state[1:176:2, ::2]
    image = image.mean(axis=2)
    image[image==color] = 0
    image = (image - 128) / 128 - 1
    image = np.expand_dims(image.reshape(88, 80, 1), axis=0)
    return image

# Define the DQN class
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9        # discount factor
        self.epsilon = 0.8      # epsilon value
        self.update_rate = 1000 # Update rate for the target network
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())
    # Build the DQN
    def build_network(self):
        # 1st Convolutional Layer
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same',
                         input_shape=self.state_size))
        model.add(Activation('relu'))
        # 2nd Convolutional Layer
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        # 3rd Convolutional Layer
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        # Flatten the feature maps & Feed them to the Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # Compile the model with loss as MSE
        model.compile(loss='mse', optimizer=Adam())
        # Return the model
        return model
    # Storing the transition information
    def store_transition(self, state, action, reward,
                         next_state, done):
        self.replay_buffer.append((state, action, reward,
                                   next_state, done))
    # Defining the epsilon-greedy policy
    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])
    # Defining the training
    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:    # If next_state is not a terminal state
                target_Q = (reward + self.gamma *
                            np.amax(self.target_network.predict(next_state)))
            else:           # If next_state is a terminal state
                target_Q = reward
            Q_values = self.main_network.predict(state)
            Q_values[0][action] = target_Q
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)
    # Updating the target network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

# Training the DQN
num_episodes = 500
num_timesteps = 20000
batch_size = 8
# Set the number of past game screens we want to consider
num_screens = 4
dqn = DQN(state_size, action_size)
done = False
time_step = 0
for i in range(num_episodes):
    Return = 0
    state = preprocess_state(env.reset())
    for t in range(num_timesteps):
        env.render()
        time_step += 1
        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()
        action = dqn.epsilon_greedy(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        dqn.store_transition(state, action, reward, next_state, done)
        state = next_state
        Return += reward
        if done:
            print("Episode: ", i, ", Return", Return)
            break
        # Only train the network
        # If the number of transitions in the replay buffer > the batch size
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)
