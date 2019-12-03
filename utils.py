import numpy as np 
import torch


# take in a list of arrays not necessarily of equal length
# takes the mean o them where shorter ones are padded with their last value
def unequal_mean(arrays):
    maxlen = np.max([len(a) for a in arrays])





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

