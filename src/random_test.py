import enviroment
import agents
from tqdm import tqdm
import numpy as np
import time

agent = agents.random_agent()

env = enviroment.SuperTicTacToe()

human = agents.human_agent()

player = 1

env.reset()



state, reward, done, info = env.step(2)

print(state)

state, reward, done, info = env.step(1)

state, reward, done, info = env.step(2)

state, reward, done, info = env.step(3)

print(state)

env.reset()

state, reward, done, info = env.step(1)

print(state)


