
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
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import Normalizer
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam

env = gym.make('Pong-v0')
# state_size = 6400
# action_size = 2
state_size = env.observation_space.shape[0]
# action_size = 2
action_size = env.action_space.n
learning_rate = 0.001
episodes = 1000
sample_batch_size = 32
ACTIONS_LABLES = [2, 3]


# model = Sequential()
# model.add(Dense(64, input_dim=state_size, activation='relu'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(action_size, activation='linear'))
# model.compile(loss='mse', optimizer=Adam(lr=learning_rate))



tsteps = 1
batch_size = 1
num_features = 4

model = Sequential()
model.add(LSTM(64,
               input_shape=(1, num_features),
               return_sequences=True,
               stateful=False))
model.add(Dropout(0.5))

model.add(LSTM(64,
               input_shape=(1, num_features),
               return_sequences=False,
               stateful=False))
model.add(Dropout(0.5))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('softmax')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer=adam)




gamma = 0.95
exploration_rate   = 1.0
exploration_min    = 0.01
exploration_decay  = 0.995
memory= deque(maxlen=2000)
A = np.array([])

def preprocess(img):
    """Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector."""
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(img.reshape(1, -1))
    img1 = img[0]
    img2 = img[1]
    img3 = img[2]
    img4 =img[3]


    xdata = np.array([[[img1, img2,img3,img4]]])
    # scaler = Normalizer().fit(img.reshape(-1, 1))
    # normalizedX = scaler.transform(img.reshape(-1, 1))
    # data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
    # data_scaled_minmax = data_scaler_minmax.fit_transform(img.reshape(-1, 1))
    return xdata

for index_episode in range(episodes):
    state = preprocess(env.reset())
    # state =env.reset()
    # state = state.reshape((1, 4, 1))
    # print(state)

    # state = np.reshape(state, [1, state_size])
    total_reward = 0

    done = False
    index = 0
    while not done:
        # print(state)
        env.render()
        if np.random.rand() <= exploration_rate:
            action = random.randrange(action_size)
        else:
            act_values = model.predict(state.reshape(1,1920))
            action = np.argmax(act_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess(next_state)
        next_state = state.reshape(1,1920)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        index += 1
        total_reward += reward

    A = np.append(A, total_reward)
    test = np.mean(A)
    print("Episode {}# Score: {} Exploration rate: {}".format(index_episode, test,exploration_rate ))
    # replay(exploration_rate,sample_batch_size)
    if len(memory) < sample_batch_size:
            print('full')
    else:
        sample_batch = random.sample(memory, sample_batch_size)
        X_train = []
        y_train = []
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            X_train.append(state)
            y_train.append(target_f.reshape(4,))

        # model.fit(state, target_f, epochs=1, verbose=0)
        X_train = np.squeeze(np.array(X_train), axis=(1))
        y_train = np.array(y_train)
        model.fit(X_train, y_train, batch_size=sample_batch_size, epochs=1, verbose=0)
        if exploration_rate > exploration_min:
            exploration_rate *= exploration_decay
            # print(exploration_rate)



# self.agent.save_model()
# model.save('cartpoolModel.h5')
