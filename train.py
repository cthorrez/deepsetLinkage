import numpy as np
import torch
from hac import HAC
from models import DeepSetLinkage
from utils import process_pair_features, find_thresh
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

    # train_blocks = ['robinson_h', 'mcguire_j']
    # val_blocks = ['moore_a', 'blum_a']
    # test_blocks = ['lee_l', 'jones_s']



    num_epochs = args['n_epochs']
    feature_dim = 14
    margin = args['margin']
    model = DeepSetLinkage(in_dim=feature_dim, lr=args['lr'], linear=args['linear'])



    train_losses = []
    val_losses = []

    prev_train_loss = np.inf
    best_val_loss = np.inf
    best_model = deepcopy(model)
    irritaion = 0

    for epoch in range(num_epochs):
        

        train_loss = 0
        for idx, tb in enumerate(train_blocks):
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(tb), delimiter=',', dtype=np.float)

            pairs = process_pair_features(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(tb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

            loss = hac.train_epoch()
            #print(tb, 'train loss:', loss)
            train_loss += loss
        train_loss = train_loss/len(train_blocks)
        print('epoch:', epoch, 'train loss:', train_loss)

  
        val_loss = 0
        for idx, vb in enumerate(val_blocks):
            pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
            pairs = process_pair_features(pair_features)
            gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
            hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)
            
            loss = hac.validate()
            #print(vb, 'val loss:', loss)
            val_loss += loss
        val_loss = val_loss/len(val_blocks)
        print('epoch:', epoch, 'val loss:', val_loss)


        if train_loss > prev_train_loss:
            print('train loss went up, stopping now')
            model = best_model
            break

        if val_loss >= best_val_loss:
            irritaion += 1
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)
            irritaion = 0

        if irritaion >= args['patience']:
            print("val loss hasn't improved in {} epochs, stopping now".format(args['patience']))
            model = best_model
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        prev_train_loss = train_loss


    np.save(args['path']+'/train_losses_'+str(seed), np.array(train_losses))
    np.save(args['path']+'/val_losses_'+str(seed), np.array(val_losses))
        

    # find f1 score
    link_list = []
    f1_list = []
    # for idx, vb in enumerate(val_blocks):
    for idx, vb in enumerate(val_blocks + train_blocks):
        pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(vb), delimiter=',', dtype=np.float)
        pairs = process_pair_features(pair_features)
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(vb), delimiter='\t', dtype=np.float)[:,1]
        hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

        links, f1s = hac.cluster()
        link_list.append(links)
        f1_list.append(f1s)
    
        idx = np.argmax(f1s)
        best_f1 = f1s[idx]
        best_link = links[idx]
        print('{} best f1: {} best link: {}'.format(vb, best_f1, best_link))


    best_thresh = find_thresh(link_list, f1_list)
    print('best threshold:', best_thresh)


    test_f1s = []
    for idx, teb in enumerate(test_blocks):
        pair_features = np.loadtxt('data/rexa/{}/pairFeatures.csv'.format(teb), delimiter=',', dtype=np.float)
        pairs = process_pair_features(pair_features)
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(teb), delimiter='\t', dtype=np.float)[:,1]
        hac = HAC(pairs, gt_clusters, model, margin=margin, use_gpu=use_gpu)

        f1 = hac.get_test_f1(best_thresh)
        print('test f1 on {}: {}'.format(teb, f1))
        test_f1s.append(f1)

    print('test f1:', np.mean(test_f1s))
        
        
def main(args, seed):
    train(args, seed)

if __name__ == '__main__':
    cfg = 'config.json'
    args = json.load(open(cfg))
    seed = int(sys.argv[1])
    
    main(args, seed)