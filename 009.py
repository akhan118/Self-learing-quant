import gym
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from IPython.display import clear_output
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam


# model = Sequential()
# model.add(Dense(175, kernel_initializer="lecun_uniform"))
# model.add(Activation('relu'))
# #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?
#
# model.add(Dense(150, kernel_initializer="lecun_uniform"))
# model.add(Activation('relu'))
# #model.add(Dropout(0.2))
#
# model.add(Dense(6, kernel_initializer="lecun_uniform"))
# model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
#
# rms = RMSprop()
# model.compile(loss='mse', optimizer=rms)



tsteps = 1
batch_size = 1
num_features = 7

model = Sequential()

model.add(LSTM(150,
               input_shape=(160,3),
               return_sequences=False))
model.add(Dropout(0.5))

# model.add(LSTM(64,
#                input_shape=( 160, 3),
#                return_sequences=False,
#                stateful=False))
# model.add(Dropout(0.5))

model.add(Dense(6, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
adam = Adam()
model.compile(loss='mse', optimizer=adam)


env = gym.make('Pong-v0')
observation = env.reset()
print(observation.shape)
qval = model.predict(observation, batch_size=1)
#
print(qval)


# gamma = 0.9 #since it may take several moves to goal, making gamma high
# epsilon = 1
# env = gym.make('Pong-v0')
# epoch = 0
# for i_episode in range(1000):
#     epoch += 1
#     status = 1
#     counter = 0
#     total_reward = 0
#     average_reward =0
#     state = env.reset()
#     while(status == 1):
#         counter += 1
#         env.render()
#         qval = model.predict(state.reshape(1,100800), batch_size=1)
#         # print(qval)
#         if (random.random() < epsilon): #choose random action
#             action = np.random.randint(0,6)
#         else: #choose best action from Q(s,a) values
#             action = (np.argmax(qval))
#         # print(observation)
#         # action = env.action_space.sample()
#         new_state, reward, done, info = env.step(action)
#         newQ = model.predict(new_state.reshape(1,100800), batch_size=1)
#         maxQ = np.max(newQ)
#         if not done: #non-terminal state
#             update = (reward + (gamma * maxQ))
#         else: #terminal state
#             update = reward
#         y = np.zeros((1,6))
#         y[:] = qval[:]
#         y[0][action] = update #target output
#         model.fit(new_state.reshape(1,100800), y, batch_size=1, epochs=1, verbose=0)
#         state = new_state
#         total_reward += update
#         average_reward = total_reward / counter
#         if done:
#             print("Episode {} ".format(epoch) + "finished after {} timesteps".format(counter) + " and with  {} average reward ".format(average_reward) )
#             # print("Episode finished with  {} average reward ".format(average_reward))
#             status = 0
#             counter = 0
#             terminal_state = 1
#
#     plt.subplot(2, 1, 1)
#     plt.plot(epoch, counter, 'o-')
#     plt.title('Epoch vs time steps')
#     plt.ylabel('timesteps')
#     plt.xlabel('epoch')
#
#     x2= epoch
#     plt.subplot(2, 1, 2)
#     plt.plot(x2, average_reward, '.-')
#     plt.title('Epoch vs average reward')
#     # plt.xlabel('epoch')
#     plt.ylabel('average reward')
#     # plt.scatter(epoch, counter)
#
#     # plt.scatter(epoch, average_reward)
# # saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
# model.save('randomModel.h5')
# plt.show()
