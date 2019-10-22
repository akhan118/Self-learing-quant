# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
from collections      import deque
import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib
from td.tdapi_test import Td
from datetime import datetime
import time
from talib import MA_Type
import talib

# import quandl

'''
Name:        Learn how to trade .

Author:      Ahmad Khan


'''

#Load data
# def read_convert_data(symbol='XBTEUR'):
#     if symbol == 'XBTEUR':
#         prices = quandl.get("BCHARTS/KRAKENEUR")
#         prices.to_pickle('data/XBTEUR_1day.pkl') # a /data folder must exist
#     if symbol == 'EURUSD_1day':
#         #prices = Quandl.get("ECB/EURUSD")
#         prices = pd.read_csv('data/EURUSD_1day.csv',sep=",", skiprows=0, header=0, index_col=0, parse_dates=True, names=['close', 'date', 'high', 'low', 'open', 'volume'])
#         prices.to_pickle('data/EURUSD_1day.pkl')
#     # print(prices)
#     return

def load_data(test=False):
    start_date ='07 3 2017 1:33PM'
    end_date = '11 3 2017 1:33PM'
    tdapi = Td()
    start_date = datetime.strptime(start_date, '%m %d %Y %I:%M%p')
    end_date = datetime.strptime(end_date, '%m %d %Y %I:%M%p')
    df=tdapi.get_price_history('F',tdapi.unix_time_millis(start_date),tdapi.unix_time_millis(end_date))
    x_train = df.iloc[-2014:-30,]
    #print(x_train.shape)
    x_test= df.iloc[-2014:,]
    #print(x_test.shape)

    # print(df.shape)
    # print(x_train.shape)
    # print(x_test.shape)
    if test:
        return x_test
    else:
        return x_train


#Initialize first state, all items are placed deterministically
def init_state(indata, test=False):
    close = indata['close'].values
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    sma15 = SMA(indata, timeperiod=15)
    sma60 = SMA(indata, timeperiod=60)
    rsi = RSI(indata, timeperiod=14)
    sma5 = SMA(indata, timeperiod=5)
    output = MOM(indata, matype=MA_Type.T3)
    upper, middle, lower = talib.BBANDS(close, matype=MA_Type.T3)
    macd_talib, signal, hist = talib.MACD(close,
                                          fastperiod=12,
                                          slowperiod=26,
                                          signalperiod=9)
    

    # #--- Preprocess data
    xdata = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi,sma5, upper, middle, lower))
    xdata = np.nan_to_num(xdata)
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
        joblib.dump(scaler, 'data/scaler.pkl')
    elif test == True:
        scaler = joblib.load('data/scaler.pkl')
        xdata = np.expand_dims(scaler.fit_transform(xdata), axis=1)
    state = xdata[0:1, 0:1, :]
    # print(xdata.shape)
    #print(state)
    #print(state.shape)

    return state, xdata, close

#indata = load_data()
#state, xdata, price_data = init_state(indata)
# Feature Scaling


#Take Action
def take_action(state, xdata, action, signal, time_step):
    #this should generate a list of trade signals that at evaluation time are fed to the backtester
    #the backtester should get a list of trade signals and a list of price data for the assett

    #make necessary adjustments to state and then return it
    time_step += 1

    #if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step + 1 == xdata.shape[0]:
        state = xdata[time_step-1:time_step, 0:1, :]
        terminal_state = 1
        signal.loc[time_step] = 0

        return state, time_step, signal, terminal_state

    #move the market data window one step forward
    state = xdata[time_step-1:time_step, 0:1, :]
    #take action
    if action == 1:
        signal.loc[time_step] = 100
    elif action == 2:
        signal.loc[time_step] = -100
    else:
        signal.loc[time_step] = 0
    #print(state)
    terminal_state = 0
    #print(signal)

    return state, time_step, signal, terminal_state

#Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    if eval == False:
         bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
         reward = bt.pnl.iloc[-1]
         
             
         if reward > 50:
             reward = 0
             reward = 1            
         if reward < 0:
             reward = 0
             reward = -1
         if reward == 0:
             reward =0
             reward = -1
           

                       
         #print(reward)
        # bt = twp.Backtest(pd.Series(data=[x for x in xdata[time_step-2:time_step]], index=signal[time_step-2:time_step].index.values), signal[time_step-2:time_step], signalType='shares')
        # reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])
        # print(reward)
        # print(xdata[time_step-2:time_step])
        # print(signal[time_step-2:time_step])
        # print(bt.data)
        # print('*******************************')
    if terminal_state == 1 and eval == True:
        #save a figure of the test set
        # print(signal)
        bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        # print(reward)

#        plt.figure(figsize=(3,4))
        bt.plotTrades()
        plt.axvline(x=400, color='black', linestyle='--')
        plt.suptitle(str(epoch))
        plt.savefig('plt1/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
        plt.close('all')

    #print(time_step, terminal_state, eval, reward)

    return reward

def evaluate_Q(eval_data, eval_model, price_data, epoch=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data,True)
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1 ,verbose=0)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True, epoch=epoch)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0

    return eval_reward

#This neural network is the the Q-function, run it like this:
#model.predict(state.reshape(1,64), batch_size=1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

tsteps = 1
batch_size = 1
num_features = 10

model = Sequential()
model.add(LSTM(128,
               input_shape=(1, num_features),
               return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(96, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(32,return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(16,return_sequences=True))
model.add(Dropout(0.2))


model.add(LSTM(8,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(4, kernel_initializer='lecun_uniform'))
model.add(Activation('softmax')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer='adam' ,metrics = ['accuracy'])



import random, timeit

start_time = timeit.default_timer()

# read_convert_data(symbol='XBTEUR') #run once to read indata, resample and convert to pickle
indata = load_data()
test_data = load_data(test=True)
epochs = 100
gamma = 0.95 #since the reward can be several time steps away, make gamma high
epsilon = 1
batchSize = 100
buffer = 200
replay = []
learning_progress = []
#stores tuples of (S, A, R, S')
h = 0
#signal = pd.Series(index=market_data.index)
signal = pd.Series(index=np.arange(len(indata)))
for i in range(epochs):
    if i == epochs-1: #the last epoch, use test data set
        indata = test_data
        signal = pd.Series(index=np.arange(len(indata)))
        state, xdata, price_data = init_state(indata, test=True)
    else:
        state, xdata, price_data = init_state(indata)
    status = 1
    terminal_state = 0
    #time_step = market_data.index[0] + 64 #when using market_data
    time_step = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        # print(state)
        qval = model.predict(state, batch_size=1)

        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4) #assumes 4 different actions
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))

        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state)
        #replay.append((state, action, reward, new_state))
        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
            #print(time_step, reward, terminal_state)
        else: #if buffer full, overwrite old values
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
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state, batch_size=1)
                newQ = model.predict(new_state, batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                if terminal_state == 0: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                #print(time_step, reward, terminal_state)
                X_train.append(old_state)
                y_train.append(y.reshape(4,))

            X_train = np.squeeze(np.array(X_train), axis=(1))
            y_train = np.array(y_train)
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)

            state = new_state
        if terminal_state == 1: #if reached terminal state, update epoch status
            status = 0
    eval_reward = evaluate_Q(test_data, model, price_data, i)
    learning_progress.append((eval_reward))
    print("Epoch #: %s Reward: %f Epsilon: %f " % (i,eval_reward, epsilon ))
    #learning_progress.append((reward))
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1.0/epochs)


elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

#model.save('SnapssStock.h5')

bt = twp.Backtest(pd.Series(data=[x[0,0] for x in xdata]), signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

#print(bt.data)
unique, counts = np.unique(filter(lambda v: v==v, signal.values), return_counts=True)
#print(np.asarray((unique, counts)).T)

plt.figure()
plt.subplot(3,1,1)
bt.plotTrades()
plt.subplot(3,1,2)
bt.pnl.plot(style='x-')
plt.subplot(3,1,3)
plt.plot(learning_progress)

plt.savefig('plt1/summary'+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
# plt.show()
