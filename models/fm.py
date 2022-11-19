import torch
import torch.nn as nn
import torch.nn.functional as F 

from .pytorch_fm.torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, FieldAwareFactorizationMachine, MultiLayerPerceptron, CompressedInteractionNetwork

class FM(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(FM, self).__init__()

        field_dims = [user_num, item_num]
        embed_dim = factor_num

        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

        

    def forward(self, user, item, features=None, logits=True):
        x = torch.cat([user.unsqueeze(1), item.unsqueeze(1)], dim=1)

        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1) if logits else torch.sigmoid(x.squeeze(1))


class FFM(torch.nn.Module):

    def __init__(self, user_num, item_num, factor_num):
        super().__init__()
        field_dims = [user_num, item_num]
        embed_dim = factor_num

        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, user, item, features, logits=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat([user.unsqueeze(1), item.unsqueeze(1)], dim=1)

        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return x.squeeze(1) if logits else torch.sigmoid(x.squeeze(1))


class DeepFM(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, user_num, item_num, factor_num, mlp_dims, dropout):
        super().__init__()
        field_dims = [user_num, item_num]
        embed_dim = factor_num

        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, user, item, features, logits=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat([user.unsqueeze(1), item.unsqueeze(1)], dim=1)

        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1) if logits else torch.sigmoid(x.squeeze(1))

import torch



class xDeepFM(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.

    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, user_num, item_num, factor_num, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        field_dims = [user_num, item_num]
        embed_dim = factor_num

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, user, item, features, logits=True):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat([user.unsqueeze(1), item.unsqueeze(1)], dim=1)

        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1) if logits else torch.sigmoid(x.squeeze(1))
