import json
import numpy as np
import matplotlib.pyplot as plt

def plot(data_path, game, name):
    with open('./'+ game + '/' + data_path) as f:
        data = json.load(f)
    for key in data:

        d = data[key]
        x = list(range(len(d)))
        x = x[0::100]
        y = d[0::100]

        plt.plot(x, y)
        plt.title(key)
        plt.xlabel('Episode')
        plt.savefig('./' + game + '/plots/' + key + '_' + name + '.png')
        plt.show()
    
if __name__ == '__main__':
    plot('data.json', 'fruitbot', '')