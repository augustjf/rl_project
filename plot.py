import json
import numpy as np
import matplotlib.pyplot as plt

def plot(data_path, name):
    with open(data_path) as f:
        data = json.load(f)
    for key in data:
        if key == 'episode_reward_history':
            reward_data = data[key][0::100]
            episodes = np.arange(0, len(reward_data)*100, 100)
            plt.plot(episodes, reward_data)
            plt.title(key)
            plt.xlabel('Episode')
            plt.savefig('./plots/' + key + '_' + name + '.png')
            plt.show()
        else:
            plt.plot(data[key])
            plt.title(key)
            plt.xlabel('Episode')
            plt.savefig('./plots/' + key + '_' + name + '.png')
            plt.show()
        
if __name__ == '__main__':
    plot('data.json', 'initial')