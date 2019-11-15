import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import itertools
from utils import dict_argmin
from eval_utils import pairwise_f1




class HAC():
    def __init__(self, pairs, gt_clusters, model, margin):
        self.pairs = pairs
        self.pair_dim = pairs[list(self.pairs.keys())[0]].shape[0]
        self.n_points = len(gt_clusters)

        self.gt_clusters = gt_clusters
        self.model = model
        self.margin = margin
 
        self.cluster_idxs = {i:[i] for i in range(self.n_points)}
        self.active_clusters = [i for i in range(self.n_points)]
        self.flat_clusters = self.get_flat_clusters()

        # print('computing initial linkage_matrix')
        self.feature_tensor = self.get_feature_tensor()
        self.linkage_matrix = self.get_linkage_matrix()

        self.next_cid = self.n_points


    def get_feature_tensor(self):
        feature_tensor = {}
        for j in range(self.n_points):
            for i in range(j):
                # i < j
                feature_tensor[i,j] = self.model.featurize(self.pairs[(i,j)].unsqueeze(0))
        return feature_tensor

    def get_linkage_matrix(self):
        #linkage_matrix = torch.zeros(self.n_points, self.n_points, requires_grad=True) 
        linkage_matrix = {}
        for j in range(self.n_points):
            for i in range(j):
                # i < j
                linkage_matrix[i,j] = self.model.score(self.feature_tensor[i,j]).squeeze()

        return linkage_matrix

    def update_linkage_matrix(self, i, j):
        # recompute linkages for new cluster i
        for cid in self.active_clusters:
            if cid == i : continue # don't compute linkage with self
            x,y = min(i,cid), max(i,cid)
            pair_features = self.get_pair_features_for_cluster_pair(x,y)
            self.linkage_matrix[i,j] = self.model.score(pair_features).squeeze()

        # set all linkages to infinite for old cluster j
        for cid in self.active_clusters:
            x,y = min(j,cid), max(j,cid)
            self.linkage_matrix[x,y] = np.inf



    def train_iter(self):
        # find clusters to merge
        # linkage_matrix = torch.triu(self.linkage_matrix)
        # linkage_matrix[linkage_matrix==0] = np.inf
        # i,j = np.unravel_index(torch.argmin(linkage_matrix), shape=self.linkage_matrix.shape)
        

        loss, done, min_pure_cluster_pair = self.get_loss()
        i,j = min_pure_cluster_pair
        # i,j = dict_argmin(self.linkage_matrix)

        if done:
            return loss, done

        # commented out because it's expensive and monotonically increasing
        # f1 = pairwise_f1(self.gt_clusters, self.flat_clusters)
        # print('merging', i, 'and', j, 'f1:', f1)
        # print('merging', i, 'and', j)


        # put things from j into i
        self.cluster_idxs[i].extend(self.cluster_idxs[j])

        # remove cluster j
        del self.cluster_idxs[j]
        self.active_clusters.remove(j)
        self.flat_clusters = self.get_flat_clusters()

        #upate the linkage matrix with values for new cluster
        self.update_linkage_matrix(i,j)

        
        return loss, done


    # returns loss and bool which tells if there are still pure cluster mergers left
    def get_loss(self):
        cluster_pairs = list(itertools.combinations(self.active_clusters, 2))
        cluster_pair_linkages = torch.FloatTensor(np.zeros(len(cluster_pairs)))
        pure_mask = torch.zeros(len(cluster_pairs), dtype=torch.bool)

        min_pure_linkage = np.inf
        min_pure_cluster_pair = None

        for cp_idx, (c1, c2) in enumerate(cluster_pairs):
            left_idxs = self.cluster_idxs[c1]
            right_idxs = self.cluster_idxs[c2]
            pure_flag = self.check_cluster_pair_pure(left_idxs, right_idxs)
            pure_mask[cp_idx] = pure_flag
            a,b = min(c1,c2), max(c1,c2)   
            cluster_pair_linkages[cp_idx] = self.linkage_matrix[a,b]

            if pure_flag:
                if self.linkage_matrix[a,b] < min_pure_linkage:
                    min_pure_linkage = self.linkage_matrix[a,b]
                    min_pure_cluster_pair = (a,b)


        # if there are no pure mergers left
        if pure_mask.sum() == 0:
            # print('no pure mergers left')
            return None, True, (0,1)

        pure_cluster_linkages = cluster_pair_linkages[pure_mask]
        dirty_cluster_linkages = cluster_pair_linkages[pure_mask==0]



        dirty_diffs = self.margin - dirty_cluster_linkages
        dirty_losses = dirty_diffs[dirty_diffs>0]

        pure_diffs = pure_cluster_linkages + self.margin
        pure_losses = pure_diffs[pure_diffs>0]

        losses = torch.cat([dirty_losses, pure_losses])
        loss = losses.mean()

        # min_pure = torch.min(pure_cluster_linkages)
        
        # dirty_diffs = min_pure - dirty_cluster_linkages
        # pure_diffs = min_pure - pure_cluster_linkages

        # dirty_losses = 


        # # loss = torch.mean(pos_diffs)
        # if (pos_diffs>0).sum() == 0:
        #     loss = 0
        # else:
        #     loss = torch.mean(pos_diffs[pos_diffs>0])



        return loss, pure_mask.sum()==0, min_pure_cluster_pair



    def check_cluster_pair_pure(self, left_idxs, right_idxs):
        all_idxs = np.array(sorted(left_idxs) + sorted(right_idxs))
        preds = self.gt_clusters[all_idxs]
        pred0 = preds[0]
        return int(np.all(preds==pred0))

    # given the ids for 2 clusters, get the features for all the crossing pairs
    def get_pair_features_for_cluster_pair(self,c1, c2):
        # get the indees of the points in each of the clusters
        left_idxs = self.cluster_idxs[c1]
        right_idxs = self.cluster_idxs[c2]

        # do their cross product
        cross_cluster_pairs = list(itertools.product(left_idxs, right_idxs))
        
        # get the features from self.pairs and put them in a big tensor
        point_pair_features = []
        for idx,(pid1, pid2) in enumerate(cross_cluster_pairs):
            a,b = min(pid1, pid2), max(pid1,pid2)
            # point_pair_features.append(self.pairs[(a,b)])
            point_pair_features.append(self.feature_tensor[a,b])

        return torch.stack(point_pair_features)

        


    def train_epoch(self):
        self.reset()
        self.model.optimizer.zero_grad()
        done = False
        epoch_loss = 0
        iterations = 0
        while not done:
            loss, done = self.train_iter()
            if (loss is None) or done : break
            epoch_loss = epoch_loss + loss 
            iterations += 1
        
        epoch_loss.backward(retain_graph=True, create_graph=True)
        self.model.optimizer.step()
        out = float(epoch_loss) / iterations
        print('epoch_loss:', out)
        return out

    def train(self):
        print('training...')
        for i in range(5):
            epoch_loss = self.train_epoch()
            print('epoch:', i, 'loss:', epoch_loss)

        torch.save(self.model, 'model')


    def get_flat_clusters(self):
        preds = np.zeros(self.n_points)
        for cid, points in self.cluster_idxs.items():
            for p in points:
                preds[p] = cid 
        return preds


    def reset(self):
        self.cluster_idxs = {i:[i] for i in range(self.n_points)}
        self.active_clusters = [i for i in range(self.n_points)]
        self.flat_clusters = self.get_flat_clusters()
        # print('resetting initial linkage_matrix')
        self.linkage_matrix = self.get_linkage_matrix()




    