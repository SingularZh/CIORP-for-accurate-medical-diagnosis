import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import MeanAggregator, Encoder


class GCNModel(nn.Module):
    def __init__(self, feat_data, feature_dim, adj_lists, gcn_out_dim):
        super(GCNModel, self).__init__()
        print("feat_data:", feat_data)
        self.features = lambda nodes: feat_data[nodes]
        self.feature_dim = feature_dim
        self.gcn_out_dim = gcn_out_dim
        self.adj_lists = adj_lists

        self.agg1 = MeanAggregator(self.features, features_dim=self.feature_dim)
        self.enc1 = Encoder(self.features, self.feature_dim, self.gcn_out_dim, self.adj_lists, self.agg1)
        self.enc2 = Encoder(self.features, self.feature_dim, self.gcn_out_dim, self.adj_lists, self.agg1)

    def forward(self, nodes):
        h_rx = self.enc1(nodes)
        h_rc = self.enc2(nodes)

        return h_rx, h_rc


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=lstm_layers)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        h_o = lstm_out[:, -1, :]

        return h_o


class TransformerModule(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, embed_dim):
        super(TransformerModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.query_dim = embed_dim // num_heads

        self.attention_linear = nn.Linear(embed_dim, 1)

        self.embedding_layer = nn.Linear(1, embed_dim)

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, features):

        features = features.unsqueeze(-1)  # [batch_size, feature_dim, 1]
        Q = torch.zeros_like(features).unsqueeze(-1)

        features = self.embedding_layer(features)  # [batch_size, feature_dim, embed_dim]
        Q = self.embedding_layer(Q)  # [batch_size, feature_dim, embed_dim]

        Q = self.query_linear(Q)  # [batch_size, feature_dim, embed_dim]
        K = self.key_linear(features)    # [batch_size, feature_dim, embed_dim]
        V = self.value_linear(features)  # [batch_size, feature_dim, embed_dim]

        # 分割成多个头
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.query_dim).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.query_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.query_dim).transpose(1, 2)

        for _ in range(self.num_layers):
            attention_weights = F.softmax(Q @ K.transpose(-2, -1) / (self.query_dim ** 0.5), dim=-1)

            Q_c = attention_weights @ V

            non_causal_attention_weights = 1 - attention_weights

            Q_nc = non_causal_attention_weights @ V

            Q = Q_c

        # 移除模拟的序列长度维度并合并多头输出
        Q_c = Q_c.transpose(1, 2).contiguous().view(Q_c.size(0), Q_c.size(2), -1)
        Q_nc = Q_nc.transpose(1, 2).contiguous().view(Q_nc.size(0), Q_nc.size(2), -1)

        # 对注意力机制的输出应用后处理线性层
        h_ox = self.attention_linear(Q_c)
        h_c = self.attention_linear(Q_nc)

        h_ox = h_ox.squeeze(2)
        h_c = h_c.squeeze(2)

        return h_ox, h_c


class CausalModel(nn.Module):
    def __init__(self, gcn_model, lstm_model, transformer_model, gcn_out_dim, hidden_dim, output_dim,
                 combine_method_xc='add'):
        super(CausalModel, self).__init__()
        self.gcn_model = gcn_model
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model

        self.gcn_out_dim = gcn_out_dim
        self.hidden_dim = hidden_dim

        self.combine_method_xc = combine_method_xc

        self.linear_c = nn.Linear(self.hidden_dim, self.hidden_dim + self.gcn_out_dim)
        self.linear_xr = nn.Linear(self.hidden_dim + 2 * self.gcn_out_dim, output_dim)
        self.linear_xc = nn.Linear(self.hidden_dim * 2 + self.gcn_out_dim if combine_method_xc == 'cat' else self.hidden_dim + self.gcn_out_dim, output_dim)


    def forward(self, gcn_data, lstm_data):
        # GCN模型提取患者自身信息特征
        h_rx, h_rc = self.gcn_model(gcn_data)

        # LSTM模型提取时间序列信息
        h_o = self.lstm_model(lstm_data)

        # Transformer模块分离因果特征和非因果特征
        h_ox, h_c = self.transformer_model(h_o)

        h_x = torch.cat((h_rx, h_ox), dim=1)

        h_r_mean = h_rc.mean(dim=0)


        random_indices = torch.randperm(h_c.size(0))
        h_c_prime = h_c[random_indices]

        h_c_prime = self.linear_c(h_c_prime)

        h_r_mean_expanded = h_r_mean.unsqueeze(0).expand(h_x.size(0), -1)

        combined_features_xr = torch.cat((h_x, h_r_mean_expanded), dim=1)

        if self.combine_method_xc == 'add':
            combined_features_xc = h_x + h_c_prime
        elif self.combine_method_xc == 'cat':
            combined_features_xc = torch.cat((h_x, h_c_prime), dim=1)
        else:
            raise ValueError("Invalid combine_method: choose 'add' or 'cat'")

        combined_features_xr = self.linear_xr(combined_features_xr)
        combined_features_xc = self.linear_xc(combined_features_xc)


        return combined_features_xr, combined_features_xc


class PretrainModel(nn.Module):
    def __init__(self, gcn_data, gcn_dim, gcn_out_dim, adj_lists, lstm_input_dim, hidden_dim, lstm_layers, num_heads, trans_layers, embed_dim, causal_dim, combine_method_xc):
        super(PretrainModel, self).__init__()

        self.gcn_model = GCNModel(feat_data=gcn_data, feature_dim=gcn_dim, adj_lists=adj_lists, gcn_out_dim=gcn_out_dim)
        self.lstm_model = LSTM(input_dim=lstm_input_dim, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
        self.transformer_model = TransformerModule(feature_dim=hidden_dim, num_heads=num_heads, num_layers=trans_layers, embed_dim=embed_dim)
        self.causal_model = CausalModel(
            gcn_model=self.gcn_model,
            lstm_model=self.lstm_model,
            transformer_model=self.transformer_model,
            gcn_out_dim=gcn_out_dim,
            hidden_dim=hidden_dim,
            output_dim=causal_dim,
            combine_method_xc=combine_method_xc
        )

        self.classifier_qc = nn.Linear(hidden_dim, 1)
        self.classifier_qnc = nn.Linear(hidden_dim, 1)

        self.causal_dim = causal_dim
        self.hidden_dim = hidden_dim
        self.gcn_out_dim = gcn_out_dim

        self.classifier_xr = nn.Linear(self.causal_dim, 1)
        self.classifier_xc = nn.Linear(self.causal_dim, 1)


    def forward(self, gcn_data, lstm_data):

        h_o = self.lstm_model(lstm_data)
        h_ox, h_c = self.transformer_model(h_o)

        zx = self.classifier_qc(h_ox)
        zc = self.classifier_qnc(h_c)

        h_xr, h_xc = self.causal_model(gcn_data, lstm_data)

        zxr = self.classifier_xr(h_xr)
        zxc = self.classifier_xc(h_xc)


        return zx, zc, zxr, zxc


class FewshotModel(nn.Module):
    def __init__(self, gcn_data, gcn_dim, gcn_out_dim, adj_lists, lstm_input_dim, hidden_dim, lstm_layers, num_heads, trans_layers, embed_dim, causal_dim, combine_method_xc):
        super(FewshotModel, self).__init__()
        self.gcn_model = GCNModel(feat_data=gcn_data, feature_dim=gcn_dim, adj_lists=adj_lists, gcn_out_dim=gcn_out_dim)
        self.lstm_model = LSTM(input_dim=lstm_input_dim, hidden_dim=hidden_dim, lstm_layers=lstm_layers)
        self.transformer_model = TransformerModule(feature_dim=hidden_dim, num_heads=num_heads, num_layers=trans_layers, embed_dim=embed_dim)
        self.causal_model = CausalModel(
            gcn_model=self.gcn_model,
            lstm_model=self.lstm_model,
            transformer_model=self.transformer_model,
            gcn_out_dim=gcn_out_dim,
            hidden_dim=hidden_dim,
            output_dim=causal_dim,
            combine_method_xc=combine_method_xc
        )

        self.combine_method_xc = combine_method_xc


    def process_support_set(self, support_set):

        class_means_xr = {}
        class_means_xc = {}

        # 首先获取单个样本的 h_xr 和 h_xc 的形状
        single_sample = support_set[0][0]
        lstm_data_single = single_sample[:-1].unsqueeze(0)
        gcn_data_single = torch.tensor([single_sample[-1, 0]], dtype=torch.int64)

        sample_h_xr, sample_h_xc = self.causal_model(gcn_data_single, lstm_data_single)

        # 初始化累加器
        h_xr_sums = [torch.zeros_like(sample_h_xr) for _ in range(2)]
        h_xc_sums = [torch.zeros_like(sample_h_xc) for _ in range(2)]

        for row in support_set:
            for class_id, sample in enumerate(row):
                lstm_data = sample[:-1].unsqueeze(0)
                gcn_data = torch.tensor([sample[-1, 0]], dtype=torch.int64)

                h_xr, h_xc = self.causal_model(gcn_data, lstm_data)

                h_xr_sums[class_id] += h_xr
                h_xc_sums[class_id] += h_xc


        # 计算 h_xr 和 h_xc 的均值
        for class_id in range(2):
            class_means_xr[class_id] = h_xr_sums[class_id] / len(support_set)
            class_means_xc[class_id] = h_xc_sums[class_id] / len(support_set)


        return class_means_xr, class_means_xc


    def compute_similarity(self, query_xr_batch, query_xc_batch, class_means_xr, class_means_xc):

        batch_similarities_xr = []
        batch_similarities_xc = []

        # 计算 h_xr 相似度
        for query_xr in query_xr_batch:
            similarities_xr = []
            for class_id in range(len(class_means_xr)):
                distance = torch.sqrt(torch.sum((query_xr - class_means_xr[class_id]) ** 2, dim=1))
                similarity = torch.exp(-distance)
                similarities_xr.append(similarity)
            batch_similarities_xr.append(torch.cat(similarities_xr, dim=-1))

        # 计算 h_xc 相似度
        for query_xc in query_xc_batch:
            similarities_xc = []
            for class_id in range(len(class_means_xc)):
                distance = torch.sqrt(torch.sum((query_xc - class_means_xc[class_id]) ** 2, dim=1))
                similarity = torch.exp(-distance)
                similarities_xc.append(similarity)
            batch_similarities_xc.append(torch.cat(similarities_xc, dim=-1))



        return torch.stack(batch_similarities_xr), torch.stack(batch_similarities_xc)
