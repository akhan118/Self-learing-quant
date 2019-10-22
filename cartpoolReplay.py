
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



for index_episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    done = False
    index = 0
    while not done:
        # print(exploration_rate)
        if np.random.rand() <= exploration_rate:
            action = random.randrange(action_size)
        else:
            act_values = model.predict(state)
            action = np.argmax(act_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        index += 1
    print("Episode {}# Score: {} Epilson: {}".format(index_episode, index + 1, exploration_rate ))
    # replay(exploration_rate,sample_batch_size)
    if len(memory) < sample_batch_size:
            print('full')
    else:
        sample_batch = random.sample(memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
        if exploration_rate > exploration_min:
            exploration_rate *= exploration_decay
            # print(exploration_rate)



# self.agent.save_model()
# model.save('cartpoolModel.h5')
