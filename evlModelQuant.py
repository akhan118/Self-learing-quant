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
import pandas as pd
from talib.abstract import *
import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)
from collections      import deque
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib
from td.tdapi_test import Td
from datetime import datetime
import time

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

def load_data(test=False):
    start_date ='01 3 2017 1:33PM'
    end_date = '01 3 2018 1:33PM'
    tdapi = Td()
    start_date = datetime.strptime(start_date, '%m %d %Y %I:%M%p')
    end_date = datetime.strptime(end_date, '%m %d %Y %I:%M%p')
    df=tdapi.get_price_history('F',tdapi.unix_time_millis(start_date),tdapi.unix_time_millis(end_date))
    x_train = df.iloc[-2000:-20,]
    x_test= df.iloc[-2000:,]
    # print(df.shape)
    # print(x_train.shape)
    # print(x_test.shape)
    if test:
        return x_test
    else:
        return x_train
    return df

def init_state(indata, test=False):
    close = indata['close'].values
    diff = np.diff(close)
    diff = np.insert(diff, 0, 0)
    sma15 = SMA(indata, timeperiod=15)
    sma60 = SMA(indata, timeperiod=60)
    rsi = RSI(indata, timeperiod=14)
    sma5 = SMA(indata, timeperiod=5)
    # #--- Preprocess data
    xdata = np.column_stack((close, diff, sma15, close-sma15, sma15-sma60, rsi,sma5))
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
    # print(state.shape)
    # print(state)

    return state, xdata, close


def get_reward(new_state, time_step, action, xdata, signal, terminal_state, eval=False, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)

    if eval == False:

        bt = twp.Backtest(pd.Series(data=[x for x in xdata[time_step-2:time_step]], index=signal[time_step-2:time_step].index.values), signal[time_step-2:time_step], signalType='shares')
        reward = ((bt.data['price'].iloc[-1] - bt.data['price'].iloc[-2])*bt.data['shares'].iloc[-1])

    if terminal_state == 1 and eval == True:
        #save a figure of the test set
        # print(signal)
        bt = twp.Backtest(pd.Series(data=[x for x in xdata], index=signal.index.values), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]
        print(bt.data)

        print('*************************************')
        plt.figure(figsize=(3,4))
        bt.plotTrades()
        plt.show()

        # plt.axvline(x=400, color='black', linestyle='--')
        # plt.text(250, 400, 'training data')
        # plt.text(450, 400, 'test data')
        # plt.suptitle(str(epoch))
        # plt.savefig('plt/'+str(epoch)+'.png', bbox_inches='tight', pad_inches=1, dpi=72)
    # print(time_step, terminal_state, eval, reward)

    return reward


# keras.models.load_model('randomModel.h5')
def evaluate_Q(eval_data, eval_model, price_data, epoch=0):
    #This function is used to evaluate the performance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata, price_data = init_state(eval_data)
    status = 1
    terminal_state = 0
    time_step = 1
    while(status == 1):
        #We start in state S
        #Run the Q function on S to get predicted reward values on all the possible actions
        qval = eval_model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        #Observe reward
        eval_reward = get_reward(new_state, time_step, action, price_data, signal, terminal_state, eval=True, epoch=epoch)
        state = new_state
        if terminal_state == 1: #terminal state
            status = 0

    return eval_reward


model = load_model('SnapStock.h5')

indata = load_data()
test_data = load_data(test=True)
state, xdata, price_data = init_state(test_data)

eval_reward = evaluate_Q(test_data, model, price_data,0)
print(eval_reward)
