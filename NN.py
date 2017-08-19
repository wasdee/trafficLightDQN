# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import keras
import numpy as np
import pandas as pd
import math


from keras.models import Sequential
from keras.layers import Dense
from random import randint


CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
# load pima indians dataset
#dataset = np.genfromtxt('fakeData.csv', delimiter=',')[1:,1:14]
#%% import data
dataset = pd.read_csv('fakeData.csv',infer_datetime_format=True).iloc[:,1:]
def to_time_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 60 *60 + int(m) * 60 + int(s)
dataset['time_sec'] = dataset.time.apply(to_time_sec)
dataset['time_sin'] = np.sin(np.pi*dataset.time_sec/43200)
dataset['time_cos'] = np.cos(np.pi*dataset.time_sec/43200)
dataset.drop(['time','time_sec'], axis=1,inplace = True)
dataset = dataset.values

#%%
"""
def importdict(filename):#creates a function to read the csv
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    return fileDATES #return the dictionary to work with it outside the function
fileDATES = importdict('time')

timebuffer = []
for i in range(1,len(fileDATES)):
    timebuffer.append((fileDATES[i]['systemtime'].split(","))) #append only time into list #A
"""
#dataset = np.array(dataset)
#for i in range(0,len(dataset)):
#    dataset[i,12] = math.sin(math.pi*randint(1,86400)/43200)

# split into input (X) and output (Y) variables
X = dataset[:,:]
Y = []
for i in range(0,len(dataset)):
    Y.append([randint(0, 1),randint(0, 1),randint(0, 1),randint(0, 1),randint(0, 1),randint(0, 1),randint(0, 1),randint(0, 1)])
Y = np.array(Y)


# create model
model = Sequential()
model.add(Dense(20, input_dim=14, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=20, batch_size=10)
predictions = model.predict(X)
#%%

##################################################################
x_t, r_0 = game_state.frame_step(do_nothing)

if args['mode'] == 'Run':
    OBSERVE = 999999999    #We keep observe, never train
    epsilon = FINAL_EPSILON
    print ("Now we load weight")
    model.load_weights("model.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print ("Weight load successfully")    
else:                       #We go to training mode
    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

t = 0
while (True):
    loss = 0
    Q_sa = 0
    action_index = 0
    r_t = 0
    a_t = np.zeros([ACTIONS])
    #choose an action epsilon greedy
    if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            q = model.predict(s_t)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1

    #We reduced the epsilon gradually
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    #run the selected action and observed next state and reward
    x_t1_colored, r_t = game_state.frame_step(a_t)

    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

    # store the transition in D
    D.append((s_t, action_index, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    #only train if done observing
    if t > OBSERVE:
        #sample a minibatch to train on
        minibatch = random.sample(D, BATCH)

        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
        print (inputs.shape)
        targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

        #Now we do the experience replay
        for i in range(0, len(minibatch)):
            state_t = minibatch[i][0]
            action_t = minibatch[i][1]   #This is action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            #terminal = minibatch[i][4]
            # if terminated, only equals reward

            inputs[i:i + 1] = state_t    #I saved down s_t

            targets[i] = model.predict(state_t)  # Hitting each buttom probability
            Q_sa = model.predict(state_t1)

            if terminal:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

        # targets2 = normalize(targets)
        loss += model.train_on_batch(inputs, targets)

    s_t = s_t1
    t = t + 1

if random.random() <= epsilon:
    print("----------Random Action----------")
    action_index = random.randrange(ACTIONS)
    a_t[action_index] = 1
else:
    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
    max_Q = np.argmax(q)
    action_index = max_Q
    a_t[max_Q] = 1









