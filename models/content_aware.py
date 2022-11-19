import torch
import torch.nn as nn
import torch.nn.functional as F 

class ContentAware(nn.Module):
    def __init__(self, 
                 user_num, 
                 item_num,
                 factor_num,
                 feature_dim
                 ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, factor_num)
        self.W = nn.Linear(feature_dim, factor_num)

    def forward(self, uid, iid, features, logits=True):
        user = self.user_embedding(uid)
        item = self.W(features.float().mean(dim=1))

        rate = (user * item).sum(1)

        return rate if logits else torch.sigmoid(rate)


