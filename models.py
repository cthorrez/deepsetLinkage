import torch
import torch.nn as nn

class DeepSetLinkage():
    def __init__(self, in_dim, lr=1e-2, linear=False, wd=0., feature_dim=14):
        super(DeepSetLinkage, self).__init__()
        
        self.in_dim = in_dim

        if linear:
            self.feature_fn = nn.Linear(in_dim, feature_dim)
            self.scoring_fn = nn.Linear(feature_dim, 1)

        else:       
            # self.feature_fn = nn.Sequential(
            #                         nn.Linear(in_dim, in_dim),
            #                         nn.ReLU(),
            #                         nn.Linear(in_dim, feature_dim)
            #                     )

            # self.scoring_fn = nn.Sequential(
            #                      nn.Linear(feature_dim, feature_dim),
            #                      nn.ReLU(),
            #                      nn.Linear(feature_dim, 1),
            #                     )
            print('nonlinear')
            self.feature_fn = nn.Sequential(
                                    nn.Linear(in_dim, feature_dim),
                                    nn.ReLU(),
                                )

            self.scoring_fn = nn.Sequential(
                                 nn.Linear(feature_dim, 1),
                                )


        params = list(self.feature_fn.parameters()) + list(self.scoring_fn.parameters()) 
        params = nn.ParameterList(params)
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    
    def featurize(self, pairs):
        x = self.feature_fn(pairs)
        # print('result of featurize', x.shape)
        return x

    def score(self, pairs):
        # input is (n,d), output is (1,1)
        mu = torch.mean(pairs, dim=0, keepdim=True)
        return self.scoring_fn(mu)


    # only use on batches of linkages between singleton clusters
    # can batch becasue don't need to take the mean when there is only 1 input
    # only called during set_linkage_matrix
    def score_batch(self, pairs):
        # input is (bs, d), output is (bs, 1)        
        x = self.scoring_fn(pairs)
        return x


    def forward(self, pairs):
        score = self.score(self.featurize(pairs))
        return score

    def to(self, device):
        self.feature_fn = self.feature_fn.to(device)
        self.scoring_fn = self.scoring_fn.to(device)
        return self
