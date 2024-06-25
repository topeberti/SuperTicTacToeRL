import enviroment
import agents
import memory
import model

from tqdm import tqdm

from keras.models import Sequential , Model
from tensorflow.keras.layers import Concatenate, Input, Conv2D, Flatten, Dense

from keras.optimizers import Adam
from keras.initializers import RandomNormal

import policies

import time

import pandas as pd

env = enviroment.SuperTicTacToe()

#get experience

def get_experience(iterations,offset):

    for i in tqdm(range(iterations)):

        dqn.move(env,i)

        mdl.learn(32)

        mem.logging_stats()

        if i != 0:

            if i % 100 == 0:
                mdl.copy_weights()
            
            if i % 1000 == 0:
                #save the model and the memory in the folder
                mem.save(mem_file)
                mdl.save(mdl_file)
                    
        

# Define the shape of the two inputs
input_shape1 = (1,) + env.observation_space.shape
input_shape2 = (1, 1)
# Define the two inputs
input1 = Input(shape=input_shape1)
input2 = Input(shape=input_shape2)
# First path
x = Conv2D(32, (3, 3), padding = 'same', activation='relu')(input1)  # Add a convolutional layer
# x = Conv2D(64, (3, 3), padding = 'same', activation='relu')(x)  # Add a convolutional layer
x = Flatten()(x)
# Concatenate the output of the first path with the second input
concatenated = Concatenate()([x, Flatten()(input2)])
# Add the final Dense layer
# output = Dense(16, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(concatenated)
# output = Dense(16, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(output)
output = Dense(9*9, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(concatenated)
# Create the model
net = Model(inputs=[input1, input2], outputs=output)
#compile the model
net.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

mem = memory.Memory(10000, r'log\memory_log.csv')

mdl = model.Model(memory = mem, model = net, future = 0.68, alfa = 0.25)

policy = policies.Probabilistic(0.03)

dqn_agent = agents.dqn_agent(model = mdl, memory = mem, policy = policy)

#load memory and model if the files exist

folder = r'resources'
mem_file = folder + r'\memory.bin'
mdl_file = folder + r'\model.h5'

try:
    mem.load(mem_file)
    mdl.load(mdl_file)
    df = pd.read_csv(r'D:\berti\Documents\Machine Learning\SuperTicTacToe\SuperTicTacToeRL\log\memory_log.csv')
    offset = len(df)
except:
    print("No memory or model or stats found")
    offset = 0

get_experience(10000,offset)

#save the model and the memory in the folder
mem.save(mem_file)
mdl.save(mdl_file)

