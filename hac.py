import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import itertools
from utils import dict_argmin
from eval_utils import pairwise_f1




class HAC():
    def __init__(self, pairs, gt_clusters, model, margin, use_gpu=False):
        self.device = torch.device('cpu')
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print('use_gpu is True but GPU is not available, using CPU')

        self.pairs = pairs
        self.pair_dim = pairs[list(self.pairs.keys())[0]].shape[0]
        self.n_points = len(gt_clusters)

        self.gt_clusters = gt_clusters
        self.model = model.to(self.device)
        self.margin = margin
 
        self.cluster_idxs = {i:[i] for i in range(self.n_points)}
        self.active_clusters = [i for i in range(self.n_points)]
        self.flat_clusters = self.get_flat_clusters()

        # print('computing initial linkage_matrix')
        self.set_feature_tensor()
        self.set_linkage_matrix()

        self.next_cid = self.n_points


    def set_feature_tensor(self):
        # feature_tensor = {}
        feature_tensor = torch.FloatTensor(np.zeros((self.n_points, self.n_points, self.pair_dim))).to(self.device)


        for j in range(self.n_points):
            for i in range(j):
                # i < j
                # feature_tensor[i,j] = self.model.featurize(self.pairs[(i,j)].to(self.device)) # dictionary version
                feature_tensor[i,j,:] = self.model.featurize(self.pairs[(i,j)].to(self.device)) # tensor version 
        
        self.feature_tensor = feature_tensor

    def set_linkage_matrix(self):
        linkage_matrix = torch.FloatTensor(np.zeros((self.n_points, self.n_points)) + np.inf).to(self.device)
        # linkage_matrix = {}
        pure_mask = torch.BoolTensor(np.zeros((self.n_points, self.n_points))).to(self.device)

        for j in range(self.n_points):
            for i in range(j):
                # i < j
                linkage_matrix[i,j] = self.model.score(self.feature_tensor[i,j].unsqueeze(0)).squeeze()
                pure_mask[i,j] = self.check_cluster_pair_pure(self.cluster_idxs[i], self.cluster_idxs[j])

        self.linkage_matrix = linkage_matrix
        self.pure_mask = pure_mask

    def update_linkage_matrix(self, i, j):
        # recompute linkages for new cluster i
        for cid in self.active_clusters:
            if cid == i : continue # don't compute linkage with self
            x,y = min(i,cid), max(i,cid)
            pair_features = self.get_pair_features_for_cluster_pair(x,y)
            self.linkage_matrix[x,y] = self.model.score(pair_features).squeeze()
            self.pure_mask[x,y] = self.check_cluster_pair_pure(self.cluster_idxs[x], self.cluster_idxs[y])

        # set all linkages to infinite for old cluster j
        # for cid in self.active_clusters:
            x,y = min(j,cid), max(j,cid)
            self.linkage_matrix[x,y] = np.inf
            self.pure_mask[x,y] = 0

        self.linkage_matrix[min(i,j), max(i,j)] = np.inf
        self.pure_mask[min(i,j), max(i,j)] = 0



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

        # if there are no pure mergers left
        if self.pure_mask.sum() == 0:
            # print('no pure mergers left')
            return None, True, (0,1)

        # cluster_pairs = list(itertools.combinations(self.active_clusters, 2))
        # cluster_pair_linkages = torch.FloatTensor(np.zeros(len(cluster_pairs))).to(self.device)
        # pure_mask = torch.zeros(len(cluster_pairs), dtype=torch.bool)

        # min_pure_linkage = np.inf
        # min_pure_cluster_pair = None
        
        # for cp_idx, (c1, c2) in enumerate(cluster_pairs):
        #     left_idxs = self.cluster_idxs[c1]
        #     right_idxs = self.cluster_idxs[c2]
        #     pure_flag = self.check_cluster_pair_pure(left_idxs, right_idxs)
        #     pure_mask[cp_idx] = pure_flag
        #     a,b = min(c1,c2), max(c1,c2)   
        #     cluster_pair_linkages[cp_idx] = self.linkage_matrix[a,b]

        #     if pure_flag:
        #         if self.linkage_matrix[a,b] < min_pure_linkage:
        #             min_pure_linkage = self.linkage_matrix[a,b]
        #             min_pure_cluster_pair = (a,b)

        # dirty_cluster_linkages = cluster_pair_linkages[pure_mask==0]

        

        active = self.linkage_matrix < np.inf
        inactive = ~active
        impure_or_inactive = ~self.pure_mask
        impure_and_active = impure_or_inactive & active
        
        

        min_pure_idx = torch.argmin(self.linkage_matrix + 1000*impure_or_inactive)
        min_pure_cluster_pair = np.unravel_index(min_pure_idx, (self.n_points, self.n_points))
        min_pure_linkage = self.linkage_matrix[min_pure_cluster_pair[0], min_pure_cluster_pair[1]]
        dirty_cluster_linkages = self.linkage_matrix[impure_and_active]


        dirty_diffs = self.margin - dirty_cluster_linkages
        dirty_loss = dirty_diffs[dirty_diffs>0].sum()
        
        min_pure_diff = min_pure_linkage + self.margin
        

        n_impure = len(dirty_cluster_linkages)
        loss = (dirty_loss + min_pure_diff)/(n_impure + 1)

        return loss, self.pure_mask.sum()==0, min_pure_cluster_pair



    def check_cluster_pair_pure(self, left_idxs, right_idxs):
        all_idxs = np.array(left_idxs + right_idxs)
        preds = self.gt_clusters[all_idxs]
        pred0 = preds[0]
        return int(np.all(preds==pred0))

    # given the ids for 2 clusters, get the features for all the crossing pairs
    def get_pair_features_for_cluster_pair(self,c1, c2):
        # get the indees of the points in each of the clusters
        left_idxs = self.cluster_idxs[c1]
        right_idxs = self.cluster_idxs[c2]

        # do their cross product
        
        # cross_cluster_pairs = list(itertools.product(left_idxs, right_idxs))
        # # get the features from self.pairs and put them in a big tensor
        # point_pair_features = []
        # for idx,(pid1, pid2) in enumerate(cross_cluster_pairs):
        #     a,b = min(pid1, pid2), max(pid1,pid2)
        #     # point_pair_features.append(self.pairs[(a,b)])
        #     point_pair_features.append(self.feature_tensor[a,b])
        # return torch.stack(point_pair_features)


        cross_cluster_pairs = torch.LongTensor(np.meshgrid(left_idxs, right_idxs)).transpose(0,2).reshape(-1,2).to(self.device)
        mins = torch.min(cross_cluster_pairs, dim=1).values
        maxes = torch.max(cross_cluster_pairs, dim=1).values
        point_pair_features = self.feature_tensor[mins, maxes,:]
        return point_pair_features


        


    def train_epoch(self):
        # self.reset()
        self.model.optimizer.zero_grad()
        done = False
        epoch_loss = 0
        iterations = 0
        print('getting epoch_loss')
        while not done:
            loss, done = self.train_iter()
            if (loss is None) or done : break
            epoch_loss = epoch_loss + loss 
            iterations += 1
        
        print("we've got to go back!")
        epoch_loss.backward(retain_graph=True, create_graph=True)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()
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




    