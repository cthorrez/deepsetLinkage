import numpy as np 
import torch


# take in a list of arrays not necessarily of equal length
# takes the mean o them where shorter ones are padded with their last value
def unequal_mean(arrays):
    maxlen = np.max([len(a) for a in arrays])
    for i in range(len(arrays)):
        pad_width = maxlen - len(arrays[i])
        arrays[i] = np.pad(arrays[i], mode='edge', pad_width=(0,pad_width))
    arrays = np.vstack(arrays)
    n, d = arrays.shape
    assert d == maxlen
    # print(arrays)
    mu = arrays.mean(axis=0)
    return mu


def find_thresh(link_list, f1_list, n=100):
    n_lists = len(link_list)
    low = np.min([np.min(l) for l in link_list])
    high= np.max([np.max(l) for l in link_list])
    x = np.linspace(start=low, stop=high, num=n)

    interpolated = np.zeros((n_lists, n))
    for idx, (links, f1s) in enumerate(zip(link_list, f1_list)):
        interpolated[idx,:] = np.interp(x, links, f1s)

    interp_mean = interpolated.mean(axis=0)
    best_thresh = x[np.argmax(interp_mean)]
    return best_thresh

def process_pair_features(pair_features):
    n_points = np.max(pair_features[:,1]).astype(np.int) + 1
    pair_dim = pair_features.shape[1] - 3
    pair_tensor = torch.FloatTensor(np.zeros((n_points, n_points, pair_dim)))
    cumsum = 0
    for i in range(n_points):
        idxs = np.arange(i+1, n_points)
        pair_tensor[i,idxs,:] = torch.FloatTensor(pair_features[cumsum:cumsum+n_points-i-1,2:-1])
        cumsum += (n_points - i -1)

    return pair_tensor

