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
	return pairs











# class lt_matrix():
# 	def __init__(self, M):
# 		self.M = M 

# 	def __getitem__(self, idxs):
# 		if len(idxs) == 1:
# 			return self.M[idxs]

# 		if len(idxs) == 2:
# 			i,j = max(idxs), min(idxs)
# 			return self.M[i,j]

# 		else:
# 			print('problem indexing lt_matrix with:', idxs)
# 			exit(1)

