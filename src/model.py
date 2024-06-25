# this class will store the model that al agents will share, it will also be the class that will train the model.

from keras.models import load_model,clone_model
import numpy as np

class Model:

    #init
    def __init__(self,memory,model = None,future =0.9,alfa=0.1):
        if model is not None:
            self.model = model
            self.future_model = clone_model(self.model)
        self.memory = memory
        self.future = future
        self.alfa = alfa
    
    #get and set methods for model and future_model
    def get_model(self):
        model_copy = clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return model_copy

    def set_model(self,model):
        self.model = model

    def get_future_model(self):
        return self.future_model

    def set_future_model(self,model):
        self.future_model = model
    
    #set methods for memory future and alfa
    def set_memory(self,memory):
        self.memory = memory

    def set_future(self,future):
        self.future = future

    def set_alfa(self,alfa):
        self.alfa = alfa
    
    #get the input shape of the model
    def get_input_shape(self):
        return self.model.input_shape
    
    
    def save(self, filename):
        #save the net
        self.model.save(filename)
    
    def load(self, filename):
        #load the net
        self.model = load_model(filename)
        self.future_model = clone_model(self.model)
        self.future_model.set_weights(self.model.get_weights())

    #training methods

    def update_q_value(self,q,reward,future_qs):
        
        fut = self.alfa * (reward - self.future * max(future_qs)) #SE RESTA EL MAXIMO DE LOS FUTUROS Qs PORQUE EL SIGUIENTE PASO LO DA EL ADVERSARIO
        nq = q * (self.alfa-1)

        return  nq+fut

    def learn(self, batch_size):

        # Retrieve experiences from memory if enough samples are available
        if len(self.memory) >= batch_size:
            episodes = self.memory.sample(batch_size)
        else:
            episodes = self.memory.sample(len(self.memory))

        experiences = []
        
        for episode in episodes:
            experiences += episode

        # Extract states, actions, rewards, and next states from experiences
        states = np.array([experience[0] for experience in experiences])
        actions = np.array([experience[1] for experience in experiences])
        rewards = np.array([experience[2] for experience in experiences])
        dones = np.array([experience[3] for experience in experiences])
        next_states = np.array([experience[4] for experience in experiences])
        previous_actions = np.array([experience[5] for experience in experiences])

        #reshape states to fit the model
        states = states.reshape(-1,1,9,9)
        previous_actions = previous_actions.reshape(-1, 1, 1, 1)
        actions = actions.reshape(-1, 1, 1, 1)
        next_states = next_states.reshape(-1,1,9,9)
        
        qs = self.model.predict([states,previous_actions],verbose=0)

        #get the future qs
        future_qs = self.future_model.predict([next_states,actions], verbose=0)

        #calculate new qs
        for i in range(len(experiences)):
            if dones[i]:
                qs[i, actions[i]] = rewards[i]
            else:
                qs[i, actions[i]] = self.update_q_value(qs[i, actions[i]], rewards[i], future_qs[i])

        #fit the model
        self.model.fit([states,previous_actions], qs, verbose=0)
    
    def copy_weights(self):
        self.future_model.set_weights(self.model.get_weights())

    def predict(self,state):
        return self.model.predict(state, verbose=0)