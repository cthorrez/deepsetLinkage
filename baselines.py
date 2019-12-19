import numpy as np
from eval_utils import pairwise_f1

def main():
    blocks = np.array(['allen_d', 'moore_a', 'lee_l', 'robinson_h',
                       'mcguire_j', 'blum_a', 'jones_s', 'young_s' ])


    one_cluster_f1s = []
    n_clusters_f1s = []
    random_f1s = []

    for block in blocks:
        gt_clusters = np.loadtxt('data/rexa/{}/gtClusters.tsv'.format(block), delimiter='\t', dtype=np.float)[:,1]
        n = len(gt_clusters)
        n_clusters = np.max(gt_clusters) + 1
        one_cluster_preds = np.zeros(n)
        n_clusters_preds = np.arange(n)
        random_preds = np.random.randint(low=0, high=n_clusters, size=n)

        one_cluster_f1s.append(pairwise_f1(gt_clusters, one_cluster_preds))
        n_clusters_f1s.append(pairwise_f1(gt_clusters, n_clusters_preds))
        random_f1s.append(pairwise_f1(gt_clusters, random_preds))



    print('one cluster f1:', np.mean(one_cluster_f1s))
    print('n clusters f1:', np.mean(n_clusters_f1s))
    print('random f1:', np.mean(random_f1s))


if __name__ == '__main__':
    main()