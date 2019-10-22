import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop



def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j


# Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4, 4, 4))
    # place player
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0])

    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4, 4, 4))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[1, 2] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))  # find grid position of player (agent)
    w = findLoc(state, np.array([0, 0, 1, 0]))  # find wall
    g = findLoc(state, np.array([1, 0, 0, 0]))  # find goal
    p = findLoc(state, np.array([0, 1, 0, 0]))  # find pit
    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridPlayer()

    return state


# Initialize grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4, 4, 4))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[randPair(0, 4)] = np.array([0, 0, 1, 0])
    # place pit
    state[randPair(0, 4)] = np.array([0, 1, 0, 0])
    # place goal
    state[randPair(0, 4)] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))
    w = findLoc(state, np.array([0, 0, 1, 0]))
    g = findLoc(state, np.array([1, 0, 0, 0]))
    p = findLoc(state, np.array([0, 1, 0, 0]))
    # If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridRand()

    return state


def makeMove(state, action):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0]))
    state = np.zeros((4, 4, 4))

    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if (new_loc != wall):
        if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0, 0, 0, 1])
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    state[wall][2] = 1
    # re-place goal
    state[goal][0] = 1

    return state


def getLoc(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j][level] == 1):
                return i, j


def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1



def dispGrid(state):
    grid = np.zeros((4, 4), dtype='<U2')
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i,j] = ' '


    if player_loc:
        grid[player_loc] = 'P'  # player
    if wall:
        grid[wall] = 'W'  # wall
    if goal:
        grid[goal] = '+'  # goal
    if pit:
        grid[pit] = '-'  # pit

    return grid


state = initGridRand()
print(dispGrid(state))
# state = makeMove(state, 1)
# state = makeMove(state, 1)
# state = makeMove(state, 1)
# state = makeMove(state, 3)
# print(dispGrid(state))
#
# print('Reward: %s' % (getReward(state),))
# dispGrid(state)


def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


model = Sequential()
model.add(Dense(164, kernel_initializer="lecun_uniform", input_shape=(64,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, kernel_initializer="lecun_uniform"))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, kernel_initializer="lecun_uniform"))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

# test = model.predict(state.reshape(1,64), batch_size=1)



from IPython.display import clear_output
import random


epochs = 50000
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):
    random_action_count = 0
    non_random_action_count = 0
    timesteps = 0
    total_reward = 0

    state = initGridRand() #using the harder state initialization function
    status = 1

    #while game still in progress
    while(status == 1):
        timesteps += 1
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,64), batch_size=1)
        # print(state)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
            random_action_count +=1
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
            non_random_action_count +=1
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)

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
                old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
                newQ = model.predict(new_state.reshape(1,64), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(64,))
                y_train.append(y.reshape(4,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=0)
            state = new_state
            total_reward += reward
        if reward != -1: #if reached terminal state, update game status
            status = 0
            # print('done')
        clear_output(wait=True)
    print("Epoch #: %s Reward: %d Epsilon: %f Timesteps: %d Random Action: %d  Q-F Actions %d" % (i,total_reward, epsilon,timesteps,random_action_count,non_random_action_count))
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1.0/epochs)


testAlgo(init=2)
model.save('randomModel.h5')
