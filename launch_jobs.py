import os
import shutil
import json
import time
import numpy as np

def main():
    args = json.load(open('config.json'))
    path = args['path']
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    shutil.copy('config.json', path)

    queues = 4*['titanx-short'] + 2*['titanx-long'] + 2*['1080ti-long'] + 4*['1080ti-short']
    for i, q in zip(np.arange(args['n_trials']), queues):
        # run trails with different seeds
        fname = path + '/results_' + str(i)
        command = 'sbatch --partition={} --gres=gpu:1 --job-name=deepset --output={} --ntasks-per-node=1 --mem=25G run.sh {}'.format(q, fname, i)
        os.system(command)
        time.sleep(1)

if __name__ == '__main__':
    main()
