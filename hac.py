import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import itertools
from eval_utils import pairwise_f1
import gc
from copy import deepcopy




class HAC():
    def __init__(self, pairs, gt_clusters, model, margin, 
                 feature_dim=14, use_gpu=False, teacher_force=1.0):
        self.device = torch.device('cpu')
        self.teacher_force = teacher_force
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                pass
                # print('use_gpu is True but GPU is not available, using CPU')

        self.pairs = pairs
        self.pair_dim = pairs.shape[2]
        self.feature_dim = feature_dim

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
        feature_tensor = torch.FloatTensor(np.zeros((self.n_points, self.n_points, self.feature_dim))).to(self.device)

        for i in range(self.n_points-1):
            idxs = np.arange(i+1,self.n_points)
            curr_pairs = self.pairs[i,idxs,:].to(self.device)
            feature_tensor[i,idxs,:] = self.model.featurize(curr_pairs)

        self.feature_tensor = feature_tensor

    def set_linkage_matrix(self):
        linkage_matrix = torch.FloatTensor(np.zeros((self.n_points, self.n_points)) + np.inf).to(self.device)
        
        gt_matrix = torch.LongTensor(self.gt_clusters).repeat(self.n_points,1).to(self.device)
        pure_mask = gt_matrix.t() == gt_matrix
        triu_mask = ~torch.tril(torch.ones(self.n_points, self.n_points, dtype=torch.bool).to(self.device))
        pure_mask = pure_mask & triu_mask

        for i in range(self.n_points-1):
            idxs = np.arange(i+1,self.n_points)
            linkage_matrix[i,idxs] = self.model.score_batch(self.feature_tensor[i,idxs,:]).squeeze()

            
        


        self.linkage_matrix = linkage_matrix
        self.pure_mask = pure_mask

    def update_linkage_matrix(self, i, j):
        feature_list = []
        idxs = np.array([x for x in self.active_clusters if x != i])
        min_idxs = np.minimum(idxs,i)
        max_idxs = np.maximum(idxs,i)

        for x,y in zip(min_idxs, max_idxs):
            pair_features = self.get_pair_features_for_cluster_pair(x,y)
            # print(pair_features.shape)
            # print(pair_features)
            feature_list.append(pair_features.mean(dim=0))
            self.pure_mask[x,y] = self.check_cluster_pair_pure(self.cluster_idxs[x], self.cluster_idxs[y])

        features = torch.stack(feature_list)
        # print('features shape', features.shape)
        scores = self.model.scoring_fn(features).squeeze()        
        self.linkage_matrix[min_idxs, max_idxs] = scores

        self.linkage_matrix[j,:] = np.inf
        self.linkage_matrix[:,j] = np.inf
        self.pure_mask[j,:] = 0
        self.pure_mask[:,j] = 0


    def train_iter(self, return_thresh=False):
        # find clusters to merge
        loss, done, min_pure_cluster_pair = self.get_loss()
        i,j = min_pure_cluster_pair
        if done: return loss, done


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

        active = self.linkage_matrix < np.inf
        inactive = ~active
        impure_or_inactive = ~self.pure_mask
        impure_and_active = impure_or_inactive & active
        
        
        min_idx = torch.argmin(self.linkage_matrix + 10000*inactive.type(torch.float))
        min_cluster_pair = np.unravel_index(min_idx.detach().cpu().numpy(), (self.n_points, self.n_points))

        min_pure_idx = torch.argmin(self.linkage_matrix + 10000*impure_or_inactive.type(torch.float))
        min_pure_cluster_pair = np.unravel_index(min_pure_idx.detach().cpu().numpy(), (self.n_points, self.n_points))
        min_pure_linkage = self.linkage_matrix[min_pure_cluster_pair[0], min_pure_cluster_pair[1]]
        dirty_cluster_linkages = self.linkage_matrix[impure_and_active]


        dirty_diffs = self.margin - dirty_cluster_linkages
        dirty_loss = dirty_diffs[dirty_diffs>0].sum()
        
        min_pure_diff = torch.clamp(min_pure_linkage + self.margin,0)
        # print(min_pure_diff)
        

        n_impure = len(dirty_cluster_linkages)
        # loss = (dirty_loss + min_pure_diff)/(n_impure + 1)
        loss = (dirty_loss/n_impure) + min_pure_diff


        # flip a coin to determine if to merge min pure or min overall
        c = np.random.rand()
        if c > self.teacher_force : 
            # not teacher forcing
            merge_pair = min_cluster_pair
        else :
            # teacher force
            merge_pair = min_pure_cluster_pair


        return loss, self.pure_mask.sum()==0, merge_pair



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

        # do their cross product using meshgrid
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
        # print('getting epoch_loss')
        while not done:
            loss, done = self.train_iter()
            # print(len(self.active_clusters))
            # if (loss is None) or done : break
            if loss is None: break
            epoch_loss = epoch_loss + loss 
            iterations += 1
        
        # print("we've got to go back!")

        # this line might be optional idk yet
        # trying not dividing
        epoch_loss = epoch_loss / iterations

        epoch_loss.backward()
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()
        out = float(epoch_loss)
        # print('epoch_loss:', out)
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
        # self.flat_clusters = self.get_flat_clusters()
        # print('resetting initial linkage_matrix')
        self.set_feature_tensor()
        self.set_linkage_matrix()



    def validate(self):
        if len(self.active_clusters) != self.n_points:
            self.reset()
        with torch.no_grad():
            done = False
            loss = 0
            iters = 0
            while not done:
                level_loss, done = self.train_iter()
                if level_loss is None : break
                loss += float(level_loss.detach().cpu())
                iters += 1
        return loss / iters


    # cluster using the given linkage function but don't train
    # does not force it to do pure clusterings
    # returns [merge_linkages], [f1scores]
    def cluster(self):
        with torch.no_grad():
            linkages = []
            f1s = []

            while len(self.active_clusters) > 2:
                merge_idx = int(torch.argmin(self.linkage_matrix).detach().cpu())
                i,j = np.unravel_index(merge_idx, (self.n_points, self.n_points))
                i,j = min(i,j), max(i,j)

                linkages.append(float(self.linkage_matrix[i,j].detach().cpu()))

                # put stuff from j into i
                self.cluster_idxs[i].extend(self.cluster_idxs[j])

                # remove cluster j
                del self.cluster_idxs[j]
                self.active_clusters.remove(j)
                self.flat_clusters = self.get_flat_clusters()

                #upate the linkage matrix with values for new cluster
                self.update_linkage_matrix(i,j)

                preds = self.get_flat_clusters()
                f1s.append(pairwise_f1(self.gt_clusters, preds))

        return np.array(linkages), np.array(f1s)



    # cluster using the given linkage function but don't train
    # does not force it to do pure clusterings. Stop when the minimum linkage available in greater than thresh
    # return the 1 score of the flat clustering at the stopping point
    def get_test_f1(self, thresh=0):
        with torch.no_grad():

            print(len(self.gt_clusters), 'data points')

            log = []

            n_merges = 0
            while len(self.active_clusters) > 2:
                min_link = torch.min(self.linkage_matrix).detach().cpu()
     

                if np.abs(min_link - thresh) < 1e-3:
                    break

                n_merges += 1
                merge_idx = int(torch.argmin(self.linkage_matrix).detach().cpu())
                merge_link = float(torch.min(self.linkage_matrix).detach().cpu())
                i,j = np.unravel_index(merge_idx, (self.n_points, self.n_points))
                i,j = min(i,j), max(i,j)
                log.append([i,j,merge_link])

                # put stuff from j into i
                self.cluster_idxs[i].extend(self.cluster_idxs[j])

                # remove cluster j
                del self.cluster_idxs[j]
                self.active_clusters.remove(j)
                self.flat_clusters = self.get_flat_clusters()

                #upate the linkage matrix with values for new cluster
                self.update_linkage_matrix(i,j)



            print(n_merges, 'merged performed')
            preds = self.get_flat_clusters()
            f1= pairwise_f1(self.gt_clusters, preds)

        return f1, np.array(log)



    
