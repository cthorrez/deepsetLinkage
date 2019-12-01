import numpy as np
import torch
from hac import HAC
from models import DeepSetLinkage
from utils import process_pair_features, single_linkage, average_linkage, complete_linkage, l2norm
import gc
import sys
import json


def train(args):
    torch.autograd.set_detect_anomaly(True)
    blocks = ['allen_d', 'moore_a', 'lee_l', 'robinson_h',
              'mcguire_j', 'blum_a', 'jones_s', 'young_s' ]

    use_gpu = args['use_gpu']

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    idxs = np.random.permutation(len(blocks))
    train_blocks = blocks[idxs[0:3]]
    val_blocks = blocks[3:5]
    test_blocks = blocks[5:8]


    num_epochs = 20
    feature_dim = 14
    margin = 2.0
    model = DeepSetLinkage(in_dim=feature_dim)
    hacs = {}
    # train_blocks = ['lee_l']
    # train_blocks = ['moore_a']
    # train_blocks = ['jones_s']
    # train_blocks = ['allen_d'] # small
    # train_blocks = ['mcguire_j'] # smallest
    # train_blocks = ['mcguire_j', 'allen_d'] # smallest

    # val_blocks = ['allen_d'] # small
    # val_blocks = ['mcguire_j'] # smallest
    # val_blocks = ['mcguire_j', 'allen_d'] # smallest

    for epoch in range(num_epochs):
        print('epoch:', epoch)
                
        for idx, tb in enumerate(train_blocks):

            print('train on', tb)
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(tb), delimiter=',', dtype=np.float)
            pairs = process_pair_features(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(tb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

            hac.train_epoch()
            

        for idx, vb in enumerate(val_blocks):
            print('validating on', vb)
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
            pairs = process_pair_features(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

            loss = hac.validate()
            print('val loss:', loss)



        

    # find f1 score
    for idx, vb in enumerate(val_blocks):
        print('finding f1 on', vb)
        pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
        pairs = process_pair_features(pair_features)
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
        hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

        links, f1s = hac.cluster()
    
        print('f1s:', f1s)




    torch.save(model, 'model')

        





def main(args):
    train(args)




if __name__ == '__main__':
    cfg = 'config.json'
    if len(sys.argv) > 1:
        cfg = sys.argv[1]

    args = json.load(open(cfg))
    
    main(args)