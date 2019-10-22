
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
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(action_size)



def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

#state = prepro(env.reset())


def evaluate_Q(eval_model):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    state = prepro(env.reset())
    total_reward = 0
    status = 1
    while(status == 1):
        env.render()

        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state.reshape(1,6400), batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, reward, done, _ = env.step(action)
        new_state = prepro(new_state)

        #Observe reward
        total_reward += reward
        state = new_state
        if done: #terminal state
            status = 0

    return total_reward

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop







num_features = 6400

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



from IPython.display import clear_output
import random

epochs = 100
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):
    timesteps = 0
    total_reward = 0
    state = prepro(env.reset())
    status = 1

    #while game still in progress
    while(status == 1):
        timesteps += 1
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        state = np.reshape(state, (state.shape[0], state.shape[1], 1))

        qval = model.predict(state, batch_size=1)
        # print(state)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,6)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, reward, done, _ = env.step(action)
        new_state = prepro(new_state)
        total_reward += reward
        env.render()

    
        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
            # print ( count2)
        else: #if buffer full, overwrite old values
            # break
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0

            replay[h] = (state, action, reward, new_state)

            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                # print(memory)
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,6400), batch_size=1)
                newQ = model.predict(new_state.reshape(1,6400), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,6))
                y[:] = old_qval[:]
                if done: #non-terminal state
                    update = reward
                else: #terminal state
                    update = (reward + (gamma * maxQ))
                y[0][action] = update
                X_train.append(old_state.reshape(6400,))
                y_train.append(y.reshape(6,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)
            state = new_state
        if done: #if reached terminal state, update game status
            status = 0
            # print('done')
        clear_output(wait=True)
    eval_reward = evaluate_Q(model)
    print("Epoch #: %s Reward: %d  RReward: %d Epsilon: %f Timesteps: %d " % (i,eval_reward,total_reward, epsilon,timesteps))
   
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1.0/epochs)

