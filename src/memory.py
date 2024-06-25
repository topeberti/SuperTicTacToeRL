#create a class named memory that will store the experiences of the agent as the keras rl sequential memroy does. 
# This class will have the following methods:

# __init__(): This method will initialize the memory as an empty list.

# append(): This method will append the experience to the memory list.

# sample(): This method will sample a batch of experiences from the memory list.

# __len__(): This method will return the number of experiences in the memory list.

import random
import pickle
import numpy as np

class Memory:
    def __init__(self,max = 5000, log_path = None):
        self.memory = []
        self.max = max
        self.log_path = log_path
        self.episode = []

    def append(self, state, action, reward, done, state1,previous_action):

        self.episode.append((state, action, reward, done, state1, previous_action))

        if done:
            self.memory.append(self.episode)
            self.episode = []

        if len(self.memory) > self.max:
            self.memory.pop(0)
        
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []

    def concat(self, memory):
        self.memory += memory.memory

    def save(self, filename):
        #save the memory to a binary
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        #load the memory from a binary
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)
    
    def logging_stats(self):

        rewards = []

        lengths = []

        wins = 0

        draws = 0

        ilegal = 0

        for episode in self.memory:

            episode_rewards = [step[2] for step in episode]
        
            rewards.append(sum(episode_rewards))

            lengths.append(len(episode_rewards))

            if episode_rewards[-1] == 10:
                wins += 1
            elif episode_rewards[-1] == -10:
                ilegal += 1
            if episode_rewards[-1] == 5:
                draws += 1

        mean_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        mean_length = np.mean(lengths)
        max_length = np.max(lengths)
        min_length = np.min(lengths)
        memory_len = len(self.memory)

        #save the stats to a file in the log path named memory_log.csv

        with open(self.log_path, 'a+') as f:
            if f.tell() == 0:
                f.write('memory_len,mean_reward,max_reward,min_reward,mean_length,max_length,min_length,wins,draws,ilegal\n')
            f.write(f'{memory_len},{mean_reward},{max_reward},{min_reward},{mean_length},{max_length},{min_length},{wins},{draws},{ilegal}')
            f.write('\n')

