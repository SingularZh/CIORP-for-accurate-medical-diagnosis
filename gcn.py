import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn import init
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self, features, features_dim):
        super(MeanAggregator, self).__init__()
        self.features = features
        self.in_features = features_dim
        self.out_features = features_dim


    def forward(self, nodes, to_neighs, num_sample=20):
        samp_neighs = [
            set(random.sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else set(to_neigh)
            for to_neigh in to_neighs
        ]

        # 构建一个独特节点列表和节点到索引的映射
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}


        # 构建邻接矩阵的mask
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))


        for i, samp_neigh in enumerate(samp_neighs):
            for n in samp_neigh:
                mask[i, unique_nodes[n]] = 1


        # 使用mask和特征矩阵进行聚合操作
        num_neigh = mask.sum(1, keepdim=True)
        num_neigh[num_neigh == 0] = 1  # 防止除以零，对没有邻居的节点进行处理
        mask = mask.div(num_neigh)  # 归一化mask
        embed_matrix = self.features(
            torch.LongTensor(unique_nodes_list))
        embed_matrix = torch.from_numpy(embed_matrix).float()
        if embed_matrix.dim() == 1:
            embed_matrix = embed_matrix.unsqueeze(0)

        # 聚合特征
        to_feats = mask.mm(embed_matrix)

        return to_feats


class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, num_sample=20):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.FloatTensor(2 * feature_dim, embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        neigh_feats_tensor = neigh_feats.float() if not isinstance(neigh_feats, np.ndarray) else torch.from_numpy(
            neigh_feats).float()
        self_feats = self.features(torch.LongTensor(nodes))
        self_feats_tensor = self_feats.float() if not isinstance(self_feats, np.ndarray) else torch.from_numpy(
            self_feats).float()

        if self_feats_tensor.dim() == 1:
            self_feats_tensor = self_feats_tensor.unsqueeze(0)

        combined = torch.cat([self_feats_tensor, neigh_feats_tensor], dim=1)
        combined = F.relu(combined.mm(self.weight))

        return combined
