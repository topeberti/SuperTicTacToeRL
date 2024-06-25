import numpy as np
import random
import os

class random_agent():

    def __init__(self, memory):
        self.name = "random_agent"
        self.memory = memory
    def move(self,env):

        state, previous_action = env.get_state()

        moves = env.possible_moves()

        x,y = random.choice(moves)

        move = x*9 + y

        state2,reward,done,info=env.step(move)

        self.remember(state,move,reward,done,state2,previous_action)

        if done:
            env.reset()

        return state2,reward,done,info
    
    def remember(self, state, action, reward, done, state1, previous_action):
        self.memory.append(state, action, reward, done, state1, previous_action)

class human_agent():

    def __init__(self):
        self.name = "human_agent"
    
    def move(self,env):

        moves = env.possible_moves()

        while True:

            #ask user to select a move in x,y format
            move = input("Select a move i x,y format: ")
            x,y = move.split(",")

            x = int(x)
            y = int(y)


            #check if the move is valid
            if (x,y) in moves:
                return x,y
            else:
                print("Invalid move")

class dqn_agent():

    def __init__(self, model,memory,policy,future =0.9,alfa=0.1, nagents = 1):
        self.name = "dqn_agent"
        self.model = model
        self.copy_model = model
        self.memory = memory
        self.policy = policy
        self.future = future
        self.alfa = alfa
        self.episodes = 0

    def move(self,env, render = False, show = False):

        state, previous_action = env.get_state()

        previous_action = np.array([previous_action])

        #reshape the state to fit the model
        state = state.reshape((-1,1,9,9))
        previous_action = previous_action.reshape(-1,1, 1, 1)

        #predict
        prediction = self.model.predict([state,previous_action])

        #reshape the prediction to a 1d array
        prediction = prediction.reshape(81)

        #get the best move
        move = self.policy.select_action(prediction,self.episodes,state = state, render = render, show = show)

        state2,reward,done,info=env.step(move)

        self.remember(state,move,reward,done,state2,previous_action)

        if done:
            env.reset()
            self.episodes += 1
        
        return state2,reward,done,info

    def remember(self, state, action, reward, done, state1, previous_action):
        self.memory.append(state, action, reward, done, state1, previous_action)
    
    #set methods for model and future_model
    def set_model(self,model):
        self.model = model
    
    def set_future_model(self,model):
        self.future_model = model
    
    def save(self, filename):


        #write in the path file
        with open(filename, 'a+') as f:
            f.write(self.name)
            f.write('\n')
            f.write(str(self.future))
            f.write('\n')
            f.write(str(self.alfa))
            f.write('\n')
            f.write(str(self.episodes))
            f.write('\n')
    
    def load(self, filename):

        #read the path file
        with open(filename, 'r') as f:
            self.name = f.readline().split('\n')[0]
            self.future = float(f.readline().split('\n')[0])
            self.alfa = float(f.readline().split('\n')[0])
            self.episodes = float(f.readline().split('\n')[0])
        

            