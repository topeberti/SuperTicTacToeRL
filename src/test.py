import enviroment
import agents
import memory
import model
import policies
import os
from tqdm import tqdm

from keras.models import Sequential , Model
from tensorflow.keras.layers import Concatenate, Input, Conv2D, Flatten, Dense

from keras.optimizers import Adam
from keras.initializers import RandomNormal

# Create the enviroment
env = enviroment.SuperTicTacToe()

mem = memory.Memory(5000,r'log\memory_log.csv')

policy = policies.Probabilistic(0.03)

# Define the shape of the two inputs
input_shape1 = (1,) + env.observation_space.shape
input_shape2 = (1, 1)
# Define the two inputs
input1 = Input(shape=input_shape1)
input2 = Input(shape=input_shape2)
x = Flatten()(input1)
# Concatenate the output of the first path with the second input
concatenated = Concatenate()([x, Flatten()(input2)])
# Add the final Dense layer
output = Dense(9*9, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(concatenated)
# Create the model
net = Model(inputs=[input1, input2], outputs=output)
#compile the model
net.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

mdl = model.Model(memory = mem, model = net, future = 0.68, alfa = 0.25)

#resources folder
folder = r'resources/test'

#delete al files in the folder
for file in os.listdir(folder):
    os.remove(os.path.join(folder, file))

random_agent = agents.random_agent(mem)

dqn_agent = agents.dqn_agent(model = mdl, memory = mem, policy = policy)

for i in tqdm(range(1000)):
    dqn_agent.move(env,i)
    if i%10==0 and i!=0:
        mdl.learn(32)
        mem.logging_stats()