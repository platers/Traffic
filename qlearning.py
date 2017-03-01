import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout

max_memory = 100000
batch_size = 64
learning_rate = 0.01
hidden_size = 32

class ExperienceReplay(object):
    def __init__(self):
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.memory = list()

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, batch_size):
        indices = random.sample(np.arange(len(self.memory)), min(batch_size,len(self.memory)) )
        miniBatch = []
        for index in indices:
            miniBatch.append(self.memory[index])
        return miniBatch


class DeepQ:
    def __init__(self, input_size, num_actions):
        self.model = self.createModel('relu', input_size, num_actions)
        self.target_model = self.createModel('relu', input_size, num_actions)

    def createModel(self, activationType, input_size, num_actions):
            layerSize = hidden_size
            model = Sequential()
            #model.add(Dense(layerSize, input_shape=(input_size, ), init='lecun_uniform'))
            model.add(Dense(num_actions, input_shape=(input_size, ), init='lecun_uniform'))
            #model.add(Activation(activationType))
            #model.add(Dense(num_actions, init='lecun_uniform'))
            model.add(Activation("linear"))
            optimizer = optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
            model.compile(loss="mse", optimizer=optimizer)
            #print model.summary()
            return model
    def getAction(self, state):
        state = np.array(state)
        qValues = self.model.predict(state.reshape(1,len(state)))[0]
        return np.argmax(qValues)
    def updateTarget(self):
        self.target_model = self.model
    def trainModel(self, batch, discount, input_size, num_actions):
        X_batch = np.empty((0, input_size), dtype = np.float64)
        Y_batch = np.empty((0, num_actions), dtype = np.float64)
        for sample in batch:
            state = np.array(sample[0][0])
            action = sample[0][1]
            reward = sample[0][2]
            newState = np.array(sample[0][3])
            isFinal = sample[1]
            qValues = self.model.predict(state.reshape(1,len(state)))[0]
            bestAction = np.argmax(self.target_model.predict(newState.reshape(1,len(newState)))[0])
            qValuesNewState = self.model.predict(newState.reshape(1,len(newState)))[0]
            targetValue = reward + discount * qValuesNewState[bestAction]

            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
            Y_sample = qValues.copy()
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
                X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward]*num_actions]), axis=0)
        return self.model.train_on_batch(X_batch, Y_batch)