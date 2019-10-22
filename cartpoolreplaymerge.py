
#encoding: utf-8

##
## cartpole.py
## Gaetan JUVIN 06/24/2017
##

import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "cartpole_weight.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:


    def run(self):

        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        learning_rate = 0.001
        episodes = 2000
        sample_batch_size = 32
        model = Sequential()
        model.add(Dense(24, input_dim=state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))


        gamma = 0.95
        exploration_rate   = 1.0
        exploration_min    = 0.01
        exploration_decay  = 0.995
        memory= deque(maxlen=2000)

        try:
            for index_episode in range(episodes):
                state = env.reset()
                state = np.reshape(state, [1,state_size])

                done = False
                index = 0
                while not done:

                    if np.random.rand() <= exploration_rate:
                        action = random.randrange(action_size)
                    else:
                        act_values = model.predict(state)
                        action= np.argmax(act_values[0])

                    next_state, reward, done, _ = env.step(action)
                    next_state = np.reshape(next_state, [1,state_size])
                    memory.append((state, action, reward, next_state, done))
                    state = next_state
                    index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                # print(len(self.agent.memory))
                # self.agent.replay(self.sample_batch_size)
                if len(memory) < 32:
                        print('full')
                else:
                    sample_batch = random.sample(memory, 32)
                    for state, action, reward, next_state, done in sample_batch:
                        target = reward
                        if not done:
                          target = reward + gamma * np.amax(model.predict(next_state)[0])
                        target_f = model.predict(state)
                        target_f[0][action] = target
                        model.fit(state, target_f, epochs=1, verbose=0)
                    if exploration_rate > exploration_min:
                        exploration_rate *= exploration_decay
                        print(exploration_rate)




        finally:
            # self.agent.save_model()
            print('completed')

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
