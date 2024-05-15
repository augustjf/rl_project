import gym
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.python.keras.backend import argmax
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.models import model_from_json


class DQN():
    def __init__(self):
        self.dropout = 0.2
        self.replay_memory_maxlen = 100000
        #deque is more efficient than list for this purpose
        self.action_history = deque([], maxlen=self.replay_memory_maxlen)
        self.state_history = deque([], maxlen=self.replay_memory_maxlen)
        self.next_state_history = deque([], maxlen=self.replay_memory_maxlen)
        self.rewards_history = deque([], maxlen=self.replay_memory_maxlen)
        self.done_history = deque([], maxlen=self.replay_memory_maxlen)


    def create_model(self, state_space, action_space):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=16, kernel_size=8, strides=4, activation='relu', data_format='channels_last', input_shape=state_space))
        model.add(layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        #model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        #model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(256))
        model.add(layers.Dense(action_space))   
        model.add(layers.Activation('softmax'))
        return model

    def memory(self, state, action, reward, next_state, done):
        self.action_history.append(action)
        self.state_history.append(state)
        self.next_state_history.append(next_state)
        self.rewards_history.append(reward)
        self.done_history.append(done)

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
        self.env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy', use_backgrounds=False, num_levels=0)
        self.n_train_episodes = 1000
        self.test_episodes = 100
        self.batch_size = 32
        self.update_target_network = 100
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.epsilon_random_steps = 10000
        self.epsilon_exploration_steps = 1000000
        self.gamma = 0.99
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001  
        self.save_weights_path = "weights.json"
        self.save_data_path = "data.json"
        self.data_dict = {}
        self.state_space = self.env.observation_space.shape
        self.action_space = 3
        self.episode_reward_history = []
        self.q_action_history = []
        self.optimizer = adam_v2.Adam(learning_rate=self.learning_rate)
        self.loss_function = Huber()
        
        self.dqn = DQN()
        self.train_dqn = self.dqn.create_model(self.state_space, self.action_space)
        self.target_dqn = models.clone_model(self.train_dqn)


    def train(self):
        episode_count = 0
        step_count = 0
        for i in range(self.n_train_episodes):
            state = self.env.reset()
            done = False
            action = 1
            episode_reward = 0

            while(not done):
                #Choose action based on epsilon greedy
                r = np.random.random()
                if step_count < self.epsilon_random_steps or r < self.epsilon:
                    action = np.random.choice(self.action_space)
                    #print('Random action: ', action)
                else:
                    action = self.find_action(state)
                
                self.epsilon -= (self.epsilon_max - self.epsilon_min)/self.epsilon_exploration_steps
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
            
                new_state, reward, done, _ = self.env.step(self.dqn.output_to_action(action))
                self.dqn.memory(state, action, reward, new_state, done)
                state = new_state
                episode_reward += reward
                step_count += 1

                if step_count > self.batch_size and step_count % 4 == 0:
                    #Sample from replay memory
                    indecies = np.random.choice(range(len(self.dqn.done_history)), self.batch_size) #Use len(done_history) to get the number of samples in the replay memory
                    state_sample = np.array(self.dqn.state_history)[indecies]
                    next_state_sample = np.array(self.dqn.next_state_history)[indecies]
                    rewards_sample = np.array(self.dqn.rewards_history)[indecies]
                    action_sample = np.array(self.dqn.action_history)[indecies]
                    done_sample = np.array(self.dqn.done_history)[indecies]

                    pred_reward = self.train_dqn(next_state_sample)
                    max_pred_rewards_ind = argmax(pred_reward, axis=1) #Can find better way to do this
                    new_q_vals = rewards_sample + self.gamma*np.array(pred_reward[0])[max_pred_rewards_ind]
                    for i, d in enumerate(done_sample): #If done, set the q value to the reward
                        if d:
                            new_q_vals[i] = rewards_sample[i]

                    mask = tf.one_hot(action_sample, self.action_space)
                    with tf.GradientTape() as tape:
                        #Calculate gradients
                        q_vals = self.train_dqn(state_sample)
                        q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)
                        self.q_action_history.append(q_action)
                        loss = self.loss_function(new_q_vals, q_action)
                    
                    grads = tape.gradient(loss, self.train_dqn.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.train_dqn.trainable_variables))

                if step_count % self.update_target_network == 0:
                    self.target_dqn.set_weights(self.train_dqn.get_weights())
                    print('Updated target network')
                
                if len(self.dqn.done_history) > self.dqn.replay_memory_maxlen:
                    #Limit the length of the replay memory
                    self.dqn.action_history.popleft(0)
                    self.dqn.state_history.popleft(0)
                    self.dqn.next_state_history.popleft(0)
                    self.dqn.rewards_history.popleft(0)
                    self.dqn.done_history.popleft(0)
                
                if done:
                    break

            episode_count += 1
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                self.episode_reward_history.pop(0)
            if episode_count % 5 == 0:
                print('Episode: ', episode_count, 'Reward: ', episode_reward)
                print('Epsilon: ', self.epsilon)
                print('Mean reward: ', np.mean(self.episode_reward_history))
                print('Step count: ', step_count)

        self.data_dict['episode_reward_history'] = self.episode_reward_history
        self.data_dict['q_action_history'] = self.q_action_history
        self.save_data_to_json()
        save_weights_to_json(self.target_dqn, self.save_path)
        
        self.env.close()

    def test(self):
        self.env = gym.make('procgen:procgen-fruitbot-v0', distribution_mode='easy', use_backgrounds=False, num_levels=0)
        state = self.env.reset()
        self.target_dqn = self.dqn.create_model(self.state_space, self.action_space)
        self.target_dqn.set_weights(load_weights_from_json(self.save_path))
        for i in range(self.test_episodes):
            action = self.find_action(state)
            new_state, reward, done, _ = self.env.step(self.dqn.output_to_action(action))
            state = new_state
        
    
    def find_action(self, state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        output_probs = self.target_dqn(state_tensor, training=False)
        action = tf.argmax(output_probs[0]).numpy()
        return action
    
    def save_data_to_json(self):
        with open(self.save_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.data_dict, f, ensure_ascii=False, indent=4)
    

def plot(data, title):
    plt.plot(data)
    plt.title(title)
    plt.show()
    plt.savefig(title + '.png')

def save_weights_to_json(model, path):
    weights_pre = model.get_weights()
    weights_dict = {}
    for i, w in enumerate(weights_pre):
        weights_dict["layer_" + str(i)] = w.tolist()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, ensure_ascii=False, indent=4)

    weights_post = load_weights_from_json(path)
    flag = 0
    for i in range(len(weights_post)):
        if not np.equal(weights_pre[i], weights_post[i]).all():
            flag = 1
    if flag:
        print('Weights not saved correctly')
    else:
        print('Weights saved correctly')


def load_weights_from_json(path):
    weights = []
    with open(path) as f:
        weights_dict = json.load(f)
    for i in range(len(weights_dict)):
        weights.append(np.array(weights_dict['layer_' + str(i)]))

    return weights



if __name__ == '__main__':  
    fruitbotDQL = EnvironmentDQL()
    fruitbotDQL.train()
    print('Training done')
    fruitbotDQL.test()

