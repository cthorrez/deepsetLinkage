import sys
import json
import numpy as np
from utils import unequal_mean
import matplotlib.pyplot as plt

def main(path):
    args = json.load(open(path+'/config.json'))
    train_arrays = [np.load(path + '/train_losses_'+str(x)) for x in range(args['n_trials'])]
    val_arrays = [np.load(path + '/val_losses_'+str(x)) for x in range(args['n_trials'])]

    train_means = unequal_mean(train_arrays)
    val_means = unequal_mean(val_arrays)

    plt.plot(train_means, color='blue', lablel='train')
    plt.plot(val_means, color='red', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path+'/training_curve.png')


if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    main(path)
