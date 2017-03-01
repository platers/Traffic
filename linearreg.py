import random
import numpy as np

learning_rate = 0.01

class Linear:
    def __init__(self, input_size, num_actions):
        self.model = [self.createModel(input_size), self.createModel(input_size)]

    def createModel(self, input_size):
            model = np.random.rand(input_size)
            return model
    def getAction(self, state):
        state = np.array(state)
        q0 = np.dot(self.model[0], state)
        q1 = np.dot(self.model[1], state)
        return int(q1 > q0)
    def trainModel(self, sa, discount, input_size, num_actions):
            state = np.array(sa[0])
            action = sa[1]
            reward = sa[2]
            newState = np.array(sa[3])
            a = self.getAction(newState)
            d = reward + discount * np.dot(self.model[a], newState) -  np.dot(self.model[action], state)
            for i in range(input_size):
                #print(self.model[action])
                self.model[action][i] += learning_rate * d * state[i]