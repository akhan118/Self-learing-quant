import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n = 10
arms = np.random.rand(n)
eps = 0.3

def reward(prob):
    reward = 0;
    for i in range(10):
        if random.random() < prob:
            reward += 1
    return reward


#initialize memory array; has 1 row defaulted to random action index
av = np.array([np.random.randint(0,(n+1)), 0]).reshape(1,2) #av = action-value
#greedy method to select best arm based on memory array (historical results)
def bestArm(a):
    # print('***********MemoryArray****************')
    # print(a)
    # print('***********MemoryArray****************')

    bestArm = 0 #just default to 0
    bestMean = 0
    for u in a:
        # print('BestArmLoop-1')
        # print(a[:,0])
        # print(u[0])
        # print(a[np.where(a[:,0] == u[0])])
        avg = np.mean(a[np.where(a[:,0] == u[0])][:, 1]) #calc mean reward for each action
        # print('avg',avg)
        # print('BestArmLoop-2')
        if bestMean < avg:
            bestMean = avg
            bestArm = u[0]
    return bestArm

plt.xlabel("Plays")
plt.ylabel("Avg Reward")
for i in range(2000):

    if random.random() > eps: #greedy arm selection
        # print(random.random())
        choice = bestArm(av)
        # print(choice)
        # print('*********************************************************************************',i)
        # print(arms)
        # print(arms[choice])

        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0)
    else: #random arm selection
        choice = np.where(arms == np.random.choice(arms))[0][0]
        # print('*********************************************************************************',i)
        # print(choice)
        thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward
        av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
    #calculate the percentage the correct arm is chosen (you can plot this instead of reward)
    # print(av[np.where(av[:,0] == np.argmax(arms))])
    percCorrect = 100*(len(av[np.where(av[:,0] == np.argmax(arms))])/len(av))
    #calculate the mean reward
    runningMean = np.mean(av[:,1])
    # print(av)
    # print(av[:,1])
    # print(runningMean)
    # print('runningMean')

    plt.scatter(i, runningMean)




plt.show()
