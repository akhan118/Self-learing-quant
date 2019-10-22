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
from keras.optimizers import Adam

# Neural Network for Deep Q Learning
# Sequential() creates the foundation of the layers.
model = Sequential()
# Input Layer of state size(4) and Hidden Layer with 24 nodes
model.add(Dense(24, input_dim=4, activation='relu'))
# Hidden layer with 24 nodes
model.add(Dense(24, activation='relu'))
# Output Layer with # of actions: 2 nodes (left, right)
model.add(Dense(2, activation='linear'))
# Create the model based on the information above
model.compile(loss='mse', optimizer=Adam(lr=0.001))


# env = gym.make('CartPole-v0')
# observation = env.reset()
#
# qval = model.predict(observation.reshape(1,4), batch_size=1)
# #
# print(qval)
# print(observation.shape)

ACTIONS_LABLES = [0,1]

gamma = 0.95 #since it may take several moves to goal, making gamma high
epsilon = 1
env = gym.make('CartPole-v0')
epoch = 1000
batchSize = 40
buffer = 80
replay = []
h=0
for i in range(epoch):
    random_action_count = 0
    non_random_action_count = 0
    status = 1
    timesteps = 0
    total_reward = 0
    average_reward =0
    state = env.reset()
    while(status == 1):
        timesteps += 1
        # env.render()
        qval = model.predict(state.reshape(1,4), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,2)
            random_action_count +=1
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
            non_random_action_count +=1
        new_state, reward, done, info = env.step(ACTIONS_LABLES[action])

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
                old_qval = model.predict(old_state.reshape(1,4), batch_size=1)
                newQ = model.predict(new_state.reshape(1,4), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,2))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(4,))
                y_train.append(y.reshape(2,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)
            state = new_state
            total_reward += reward
        if done:
            # print("Episode {} ".format(i) + "finished after {} timesteps".format(timesteps) + " and with  {} average reward ".format(total_reward) )
            # print("Episode finished with  {} average reward ".format(average_reward))
            status = 0
            terminal_state = 1
    print("Epoch #: %s Reward: %d Epsilon: %f Timesteps: %d Random Action: %d  Q-F Actions %d" % (i,total_reward, epsilon,timesteps,random_action_count,non_random_action_count))
    if epsilon > 0.1:
        epsilon -= (1.0/epoch)


    # plt.subplot(2, 1, 1)
    # plt.plot(epoch, timesteps, 'o-')
    # plt.title('Epoch vs time steps')
    # plt.ylabel('timesteps')
    # plt.xlabel('epoch')
    #
    # x2= epoch
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, average_reward, '.-')
    # plt.title('Epoch vs average reward')
    # # plt.xlabel('epoch')
    # plt.ylabel('average reward')
    # # plt.scatter(epoch, timesteps)

    # plt.scatter(epoch, average_reward)
model.save('randomModel.h5')
# plt.show()
