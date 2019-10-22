from __future__ import print_function
from pprint import pprint

import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt

from sklearn import metrics, preprocessing

'''
Name:        The Self Learning Quant, Example 1

Author:      Ahmad Khan

Created:     08/03/2018
Licence:     BSD

Requirements:
Numpy
Pandas
MatplotLib
scikit-learn
Keras, https://keras.io/
backtest.py from the TWP library. Download backtest.py and put in the same folder

/plt create a subfolder in the same directory where plot files will be saved

'''

#Load data
def load_data():
    price = np.arange(200/10.0) #linearly increasing prices
    return price

#Initialize first state, all items are placed deterministically
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

#Take Action
def take_action(state, xdata, action, signal, time_step):
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett

    #make necessary adjustments to state and then return it
    time_step += 1

    #if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step == xdata.shape[0]:
        state = xdata[time_step-1:time_step, :]
        terminal_state = 1
        signal.loc[time_step] = 0
        return state, time_step, signal, terminal_state

    #move the market data window one step forward
    state = xdata[time_step-1:time_step, :]
    #take action
    print('time-step', time_step)
    if action != 0:
        if action == 1:
            signal.loc[time_step] = 100
        elif action == 2:
            signal.loc[time_step] = -100
        elif action == 3:
            signal.loc[time_step] = 0
    terminal_state = 0

    return state, time_step, signal, terminal_state

#Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, epoch=0):
    # counter = 0;
    # counter +=1;
    # # print(counter)
    print('*******Get_reward********')
    reward = 0
    print('time_step',time_step)
    signal.fillna(value=0, inplace=True)
    if terminal_state == 0:
        #get reward for the most current action
        if signal[time_step] != signal[time_step-1] and terminal_state == 0:
            i=1
            print('signal[time_step]',signal[time_step])
            print('SIGNALsignal[time_step-1]',signal[time_step-1])
            print('SIGNAL',signal)

            while signal[time_step-i] == signal[time_step-1-i] and time_step - 1 - i > 0:
                print('time_step-1-i',time_step-1-i)
                i += 1
                print('*****i*****',i)
            reward = (xdata[time_step-1, 0] - xdata[time_step - i-1, 0]) * signal[time_step - 1]*-100 + i*np.abs(signal[time_step - 1])/10.0
            print('xdata',xdata)
            print('xdata[time_step-1, 0]',xdata[time_step-1, 0])
            print('xdata[time_step - i-1, 0]',xdata[time_step - i-1, 0])
            print('(xdata[time_step-1, 0] - xdata[time_step - i-1, 0])',(xdata[time_step-1, 0] - xdata[time_step - i-1, 0]))
            print('signal[time_step - 1]',signal[time_step - 1])
            print('signal[time_step - 1]*-100',signal[time_step - 1]*-100)
            print('(-)*signal',(xdata[time_step-1, 0] - xdata[time_step - i-1, 0]) * signal[time_step - 1]*-100)
            print('i*np.abs(signal[time_step - 1])', i*np.abs(signal[time_step - 1]))

        if signal[time_step] == 0 and signal[time_step - 1] == 0:
            reward -= 10
            # if you didn't trade for two days, you get a negative -10
    #calculate the reward for all actions if the last iteration in set
    if terminal_state == 1:
        #run backtest, send list of trade signals and asset data to backtest function
        bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        # print(bt.data)
        # print(reward)
    print('*******Get_reward********')
    return reward

def evaluate_Q(eval_data, eval_model):
    #This function is used to evaluate the perofrmance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    # print(signal)
    state, xdata = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state.reshape(1,2), batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, xdata, signal, terminal_state, i)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0
    return eval_reward

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
epochs = 40
gamma = 0.9 #a high gamma makes a long term reward more valuable
epsilon = 1
learning_progress = []
#stores tuples of (S, A, R, S')
h = 0
signal = pd.Series(index=np.arange(len(indata)))
for i in range(epochs):

    state, xdata = init_state(indata)
    print('State At Main Loop***************', state)
    print('xdata***************', xdata)

    pprint(signal)
    # pprint(xdata)
    status = 1
    terminal_state = 0
    time_step = 1
    #while learning is still in progress
    # pprint('there start')
    j =0
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        j += 1
        print('********************************* Start **********************************',j)

        # pprint('there2')
        # print('state = ',state.reshape(1,2))
        # pprint('there4')
        qval = model.predict(state.reshape(1,2), batch_size=1)
        # pprint('Prediction1')
        # print('qval',qval)
        # # pprint((np.argmax(qval)))
        # pprint('Prediction2')

        print('random',random.random())
        if (random.random() < epsilon) and i != epochs - 1: #maybe choose random action if not the last epoch
            action = np.random.randint(0,4) #assumes 4 different actions
            print('random')
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
            # Returns the indices of the maximum values along an axis.
            #
            print('argmax')

        print('action',action)
        #Take action, observe new state S'

        # print('Before Take Action')
        # print('signal before action ',signal)
        # print(state)
        # print('After Start - Before End')
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # print('new state = ',new_state)
        # print(time_step)
        # print('signal after action ',signal)
        # print(terminal_state)
        # print('After Take Action End')

        #Observe reward
        reward = get_reward(new_state, time_step, action, xdata, signal, terminal_state, i)
        #Get max_Q(S',a)
        # print('signal after reward ',signal)
        # print('reward',reward)
        #
        # newQ = model.predict(new_state.reshape(1,2), batch_size=1)
        # print('new Q =', newQ)
        # maxQ = np.max(newQ)
        # print('maxQ  =', maxQ)
        # y = np.zeros((1,4))
        # print('qval',qval)
        # y[:] = qval[:]
        # print('y before update',y)
        # print('action',action)
        # print('gamma' , gamma)
        # print('game * maxQ',(gamma * maxQ))
        if terminal_state == 0: #non-terminal state
            update = (reward + (gamma * maxQ))
        else: #terminal state (means that it is the last state)
            update = reward
        y[0][action] = update #target output
        # print('update',update)
        # print('y after update',y)
        # print('X =     ',state.reshape(1,2))
        model.fit(state.reshape(1,2), y, batch_size=1, epochs=1, verbose=0)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0
        print('********************************* end **********************************',j)

    eval_reward = evaluate_Q(indata, model)
    print("Epoch #: %s Reward: %f Epsilon: %f" % (i,eval_reward, epsilon))
    print('xdata = ',xdata)
    print('signal',signal)
    bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
    print(bt.data)
    # plt.figure()
    # bt.plotTrades()
    # plt.suptitle('epoch' + str(i))
    # plt.savefig('plt/final_trades'+'.png', bbox_inches='tight', pad_inches=1, dpi=72) #assumes there is a ./plt dir
    # plt.close('all')
    # print(bt.plotTrades())
    # plt.figure()
    # plt.subplot(3,1,1)
    # bt.plotTrades()
    # plt.subplot(3,1,2)
    # bt.pnl.plot(style='x-')
    # plt.subplot(3,1,3)
    # print('learning progress',learning_progress)
    # plt.plot(learning_progress)
    # plt.show()

    learning_progress.append((eval_reward))
    if epsilon > 0.1:
        epsilon -= (1.0/epochs)

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

#plot results
# print(signal)
bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
# print(bt.data)
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

# print(bt.data)

plt.figure()
bt.plotTrades()
plt.suptitle('epoch' + str(i))
plt.savefig('plt/final_trades'+'.png', bbox_inches='tight', pad_inches=1, dpi=72) #assumes there is a ./plt dir
plt.close('all')

plt.figure()
plt.subplot(3,1,1)
bt.plotTrades()
plt.subplot(3,1,2)
bt.pnl.plot(style='x-')
plt.subplot(3,1,3)
plt.plot(learning_progress)

plt.show()
