import numpy as np
import torch
from hac import HAC
from models import DeepSetLinkage
from utils import process_pair_features, single_linkage, average_linkage, complete_linkage, l2norm
import gc


def train():
    blocks = ['allen_d', 'moore_a', 'lee_l', 'robinson_h',
              'mcguire_j', 'blum_a', 'jones_s', 'young_s' ]

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.permutation(len(blocks))
    train_blocks = blocks[0:3]
    val_blocks = blocks[3:5]
    test_blocks = blocks[5:8]


    num_epochs = 5
    feature_dim = 14
    margin = 2.0
    model = DeepSetLinkage(in_dim=feature_dim)
    hacs = {}
    # train_blocks = ['lee_l']
    train_blocks = ['moore_a']
    # train_blocks = ['jones_s']

    for epoch in range(num_epochs):
        print('epoch:', epoch)
                
        for idx, tb in enumerate(train_blocks):

            print('block:', tb)
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(tb), delimiter=',', dtype=np.float)
            pairs = process_pair_features(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(tb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin)

            hac.train_epoch()
            
            del hac 
            gc.collect()

            # if idx == 2:
            #     break

        





def main():
    train()




if __name__ == '__main__':
    main()