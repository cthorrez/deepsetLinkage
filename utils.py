import numpy as np 
import torch
from scipy.spatial.distance import cdist


def dict_argmin(d):
    argmin = None
    cur_min = np.inf
    for k,v in d.items():
        if v < cur_min:
            cur_min = v
            argmin = k
    return argmin


def l2norm(x):
    return -x.dot(x)

def single_linkage(pairs, dist_fn):
    dists = np.array([dist_fn(p) for p in pairs])
    return np.min(dists)

def average_linkage(pairs, dist_fn):
    dists = np.array([dist_fn(p) for p in pairs])
    return np.mean(dists)

def complete_linkage(pairs, dist_fn):
    dists = np.array([dist_fn(p) for p in pairs])
    return np.max(dists)




# creates a dictionary (i,j) -> pair feature vector
def process_pair_features(pair_features):
    pairs = {}
    for row in pair_features:
        i, j = row[:2]
        i, j = int(i), int(j)
        pairs[(i,j)] = torch.tensor(row[2:-1], dtype=torch.float, requires_grad=True)
        # pairs[(i,j)] = torch.FloatTensor(row[2:-1])
    return pairs


def process_pair_features2(pair_features):
    n_points = np.max(pair_features[:,1]).astype(np.int) + 1
    pair_dim = pair_features.shape[1] - 3
    pair_tensor = torch.FloatTensor(np.zeros((n_points, n_points, pair_dim)))
    cumsum = 0
    for i in range(n_points):
        idxs = np.arange(i+1, n_points)
        pair_tensor[i,idxs,:] = torch.FloatTensor(pair_features[cumsum:cumsum+n_points-i-1,2:-1])
        cumsum += (n_points - i -1)

    return pair_tensor




# class lt_matrix():
#   def __init__(self, M):
#       self.M = M 

#   def __getitem__(self, idxs):
#       if len(idxs) == 1:
#           return self.M[idxs]

#       if len(idxs) == 2:
#           i,j = max(idxs), min(idxs)
#           return self.M[i,j]

#       else:
#           print('problem indexing lt_matrix with:', idxs)
#           exit(1)

