import gym
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.optimizers import adam_v2
from collections import deque
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizers import adam_v2


class DQN():
    def __init__(self):
        self.dropout = 0.2
        self.replay_memory_maxlen = 1000
        self.replay_memory = deque([], maxlen=self.replay_memory_maxlen)

    def create_model(self, state_space, action_space):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', data_format='channels_last', input_shape=state_space))
        model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(100))
        model.add(layers.Dense(action_space))   
        model.add(layers.Activation('softmax'))
        model.compile(optimizer=adam_v2.Adam(), loss=MeanSquaredError) #optimizer might be wrong
        return model

    def memory(self, state, action, reward, next_state):
        self.replay_memory.append((state, action, reward, next_state))

    def output_to_action(self, output):
        if output == 0:
            return 1 #LEFT
        elif output == 1:
            return 4 #DO NOTHING
        elif output == 2:
            return 7 #RIGHT
        else:
            print('Invalid output')


class EnvironmentDQL():
    def __init__(self):
        self.n_episodes = 10000
        self.epsilon = 1
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.999


    def train(self):
        env = gym.make('procgen:procgen-fruitbot-v0', render_mode='human')
        state_space = env.observation_space.shape
        action_space = 3

        dqn = DQN()
        train_dqn = dqn.create_model(state_space, action_space)
        target_dqn = keras.models.clone_model(train_dqn)

        for i in range(self.n_episodes):
            state = env.reset()
            done = False

            action = 1
            while(not done):
                #Choose action based on epsilon greedy
                r = np.random.random()
                if r < self.epsilon:
                    action = np.random.choice(action_space)
                    print('Random action: ', action)
                else:
                    action = train_dqn(state)
                    print('DQN action: ', action)
                
                self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

                new_state, reward, done, _ = env.step(dqn.output_to_action(action))
                dqn.memory.append((state, action, reward, new_state))
                state = new_state






if __name__ == '__main__':  
    fruitbotDQL = EnvironmentDQL()
    fruitbotDQL.train()

