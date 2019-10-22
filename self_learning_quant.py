from __future__ import print_function
from pprint import pprint

import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt

from sklearn import metrics, preprocessing




#Load data
def load_data():
    price = np.arange(200/10.0) #linearly increasing prices
    return price

def init_state(data):
    close = data
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)
    #--- Preprocess data
    xdata = np.column_stack((close, diff))
    # pprint(xdata)
    xdata = np.nan_to_num(xdata)
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(xdata)
    # pprint(xdata)
    state = xdata[0:1, :]
    # pprint(state)
    return state, xdata





#This neural network is the the Q-function, run it like this:
#model.predict(state.reshape(1,64), batch_size=1)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop





model = Sequential()
model.add(Dense(4, init='lecun_uniform', input_shape=(2,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout in this example

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

import random, timeit

start_time = timeit.default_timer()
indata = load_data()
epochs = 2

for i in range(epochs):

    state, xdata = init_state(indata)
    status = 1
    terminal_state = 0
    time_step = 1
    #while learning is still in progress
    while(status < 4):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        status +=1
        print('data =',state.reshape(1,2))
        qval = model.predict(state.reshape(1,2), batch_size=1)
        print('prediction =',qval)
        action = (np.argmax(qval))
        print('action',action)
        y = np.zeros((1,4))
        y[:] = qval[:]
        y[0][action] += 1 #target output
        print('Y=',y)
        model.fit(state.reshape(1,2), y, batch_size=1, epochs=1, verbose=0)


    # eval_reward = evaluate_Q(indata, model)
    print("Epoch #: %s Reward: %f Epsilon: %f" )


print("Completed in %f")
