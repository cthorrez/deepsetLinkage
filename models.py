import torch
import torch.nn as nn

class DeepSetLinkage():
    def __init__(self, in_dim):
        super(DeepSetLinkage, self).__init__()
        
        self.in_dim = in_dim
        self.feature_fn = nn.Sequential(
                                nn.Linear(in_dim, in_dim),
                                nn.ReLU(),
                                nn.Linear(in_dim, in_dim)
                            )

        self.scoring_fn = nn.Sequential(
                             nn.Linear(in_dim, in_dim),
                             nn.ReLU(),
                             nn.Linear(in_dim, 1),
                            )

        # self.feature_fn = nn.Sequential(
        #                         nn.Linear(in_dim, in_dim)
        #                     )

        # self.scoring_fn = nn.Sequential(
        #                      nn.Linear(in_dim, 1)
        #                     )

        params = list(self.feature_fn.parameters()) + list(self.scoring_fn.parameters()) 
        params = nn.ParameterList(params)
        self.optimizer = torch.optim.Adam(params, lr=1e-2)
    
    def featurize(self, pairs):
        return self.feature_fn(pairs)

    def score(self, pairs):
        mu = torch.mean(pairs, dim=0, keepdim=True)
        return self.scoring_fn(mu)

    def forward(self, pairs):
        score = self.score(self.featurize(pairs))
        return score

    def to(self, device):
        self.feature_fn = self.feature_fn.to(device)
        self.scoring_fn = self.scoring_fn.to(device)
        return self
