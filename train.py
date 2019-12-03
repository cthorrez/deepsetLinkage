import numpy as np
import torch
from hac import HAC
from models import DeepSetLinkage
from utils import process_pair_features, process_pair_features2
import sys
import json
import time
from copy import deepcopy


def train(args, seed=0):
    blocks = np.array(['allen_d', 'moore_a', 'lee_l', 'robinson_h',
              'mcguire_j', 'blum_a', 'jones_s', 'young_s' ])

    use_gpu = args['use_gpu']


    np.random.seed(seed)
    torch.manual_seed(seed)
    idxs = np.random.permutation(len(blocks))
    train_blocks = blocks[idxs[0:3]]
    val_blocks = blocks[idxs[3:5]]
    test_blocks = blocks[idxs[5:8]]



    num_epochs = args['n_epochs']
    feature_dim = 14
    margin = args['margin']
    model = DeepSetLinkage(in_dim=feature_dim, lr=args['lr'], linear=args['linear'])



    train_losses = []
    val_losses = []

    prev_train_loss = np.inf
    prev_val_loss = np.inf
    prev_model = deepcopy(model)
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
            #print(tb, 'train loss:', loss)
            train_loss += loss
        print('epoch:', epoch, 'train loss:', train_loss/len(train_blocks))

  
        val_loss = 0
        for idx, vb in enumerate(val_blocks):
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
            pairs = process_pair_features2(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)
            
            loss = hac.validate()
            #print(vb, 'val loss:', loss)
            val_loss += loss
        print('epoch:', epoch, 'val loss:', val_loss/len(val_blocks))


        if train_loss > prev_train_loss:
            print('train loss went up, stopping now')
            model = prev_model
            break

        if val_loss > prev_val_loss:
            print('val loss went up, stopping now')
            model = prev_model
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        prev_model = deepcopy(model)
        prev_train_loss = train_loss
        prev_val_loss = val_loss



        

    # find f1 score
    for idx, vb in enumerate(val_blocks):
        print('finding f1 on', vb)
        pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
        pairs = process_pair_features2(pair_features)
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
        hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

        links, f1s = hac.cluster()
    
        idx = np.argmax(f1s)
        print('best f1:', f1s[idx])
        print('linkage for best f1:', links[idx])
        



    np.save(args['path']+'/train_losses_'+str(seed), np.array(train_losses))
    np.save(args['path']+'/val_losses_'+str(seed), np.array(val_losses))


    torch.save(model, args['path']+'/model')

        
def main(args, seed):
    train(args, seed)

if __name__ == '__main__':
    cfg = 'config.json'
    args = json.load(open(cfg))
    seed = int(sys.argv[1])
    
    main(args, seed)