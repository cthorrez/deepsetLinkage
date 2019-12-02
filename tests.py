import numpy as np
import torch
from hac import HAC
from models import DeepSetLinkage
from utils import process_pair_features, process_pair_features2
import gc
import sys
import json
import time


def train(args):
    torch.autograd.set_detect_anomaly(True)
    blocks = np.array(['allen_d', 'moore_a', 'lee_l', 'robinson_h',
              'mcguire_j', 'blum_a', 'jones_s', 'young_s' ])

    use_gpu = args['use_gpu']

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    idxs = np.random.permutation(len(blocks))
    train_blocks = blocks[idxs[0:3]]
    val_blocks = blocks[idxs[3:5]]
    test_blocks = blocks[idxs[5:8]]



    num_epochs = 10
    feature_dim = 14
    margin = 2.0
    model = DeepSetLinkage(in_dim=feature_dim)

    # train_blocks = ['lee_l']
    # train_blocks = ['moore_a']
    # train_blocks = ['jones_s']
    # train_blocks = ['allen_d'] # small
    # train_blocks = ['mcguire_j'] # smallest
    # train_blocks = ['mcguire_j', 'allen_d', 'robinson_h', 'moore_a'] # smallest

    # val_blocks = ['lee_l'] # small
    # val_blocks = ['mcguire_j'] # smallest
    # val_blocks = ['mcguire_j', 'allen_d'] # smallest

    for epoch in range(num_epochs):
                        
        train_loss = 0
        for idx, tb in enumerate(train_blocks):
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(tb), delimiter=',', dtype=np.float)
            # print(tb)
            # start = time.time()
            # pairs = process_pair_features(pair_features)
            # print('og time:', time.time() - start)

            # start = time.time()
            pairs = process_pair_features2(pair_features)
            # print('new time:', time.time() - start)

            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(tb), delimiter='\t', dtype=np.float)[:,1]


            # n_points = len(gt_clusters)
            # for i in range(n_points):
            #     for j in range(i+1, n_points):
            #         assert((pairs[(i,j)] - pairs2[i,j,:]).sum() == 0)

            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

            loss = hac.train_epoch()
            print(tb, 'train loss:', loss)
            train_loss += loss
        print('epoch:', epoch, 'train loss:', train_loss/len(train_blocks))

            
        val_loss = 0
        for idx, vb in enumerate(val_blocks):
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
            pairs = process_pair_features2(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)
            
            loss = hac.validate()
            print(vb, 'val loss:', loss)
            val_loss += loss
        print('epoch:', epoch, 'val loss:', val_loss/len(val_blocks))



        

    # find f1 score
    for idx, vb in enumerate(val_blocks):
        print('finding f1 on', vb)
        pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
        pairs = process_pair_features2(pair_features)
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
        hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

        links, f1s = hac.cluster()
    
        print('links:', links)
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