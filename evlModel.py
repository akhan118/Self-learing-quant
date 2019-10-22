import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model = load_model('randomModel.h5')
# keras.models.load_model('randomModel.h5')


env = gym.make('CartPole-v0')
ACTIONS_LABLES = [0,1]
def testAlgo(init=0):
    i = 0
    state = env.reset()
    status = 1
    total_reward = 0
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,4), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        new_state, reward, done, info = env.step(ACTIONS_LABLES[action])
        # env.render()
        state = new_state
        total_reward += reward
        if done:
            status = 0
            print("Reward: %s" % (total_reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 4000):
            print("Game lost; too many moves.")
            break

testAlgo()
