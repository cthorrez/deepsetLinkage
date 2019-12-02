import os

def main():
    command = 'sbatch --partition=1080ti-short --gres=gpu:1 --job-name=deepset_clustering --output=bash_test.out --ntasks-per-node=1 --mem=25G run.sh'

if __name__ == '__main__':
    main()