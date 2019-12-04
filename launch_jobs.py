import os
import shutil
import json
import time

def main():
    args = json.load(open('config.json'))
    path = args['path']
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    shutil.copy('config.json', path)

    
    for i in range(args['n_trials']):
        # run trails with different seeds
        fname = path + '/results_' + str(i)
        command = 'sbatch --partition=1080ti-short --gres=gpu:1 --job-name=deepset --output={} --ntasks-per-node=1 --mem=25G run.sh {}'.format(fname, i)
        os.system(command)
        time.sleep(1)

if __name__ == '__main__':
    main()