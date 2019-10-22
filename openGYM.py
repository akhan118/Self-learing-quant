import gym ,sys,numpy as np
import numpy as np
np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)

import pandas as pd
import backtest as twp
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from talib.abstract import *
from sklearn.externals import joblib

env = gym.make('CartPole-v1')
print(env.observation_space.shape[0])

for i_episode in range(2):
    observation = env.reset()
    for t in range(1):
        # env.render()
        # print(observation.shape)
        # print(observation.size)
        print(observation)

        scaler = preprocessing.StandardScaler()
        xdata = scaler.fit_transform(observation.reshape(1, -1))
        img1 = observation[0]
        img2 = observation[1]
        img3 = observation[2]
        img4 =observation[3]


        xdata = np.array([[[img1, img2,img3,img4]]])
        # xdata = np.nan_to_num(xdata)
        print(xdata)



        print(xdata)
        print(xdata.shape)
        print(xdata.size)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
