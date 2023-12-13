import math

import torch
import torch.nn as nn

from cogdl.utils import edge_softmax, get_activation
from cogdl.utils.spmm_utils import MultiHeadSpMM


class myGATConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        num_etypes,
        in_features,
        out_features,
        nhead,
        device,
        feat_drop=0.0,
        attn_drop=0.5,
        negative_slope=0.1,
        residual=False,
        activation=None,
        alpha=0.05,
    ):
        super(myGATConv, self).__init__()
        self.device = device
        self.edge_feats = edge_feats   #边的特征，按照边的类型给每种边一个embedding
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.edge_emb = nn.Parameter(torch.zeros(size=(num_etypes, edge_feats)))  # nn.Embedding(num_etypes, edge_feats)给边赋特征，每条边有对应的特征向量

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features * nhead)).to(self.device)  #节点的权重矩阵
        self.W_e = nn.Parameter(torch.FloatTensor(edge_feats, edge_feats * nhead)).to(self.device)   #边的权重矩阵

        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_features))).to(self.device)
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_features))).to(self.device) #节点矩阵
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_feats))).to(self.device) #边矩阵

        self.mhspmm = MultiHeadSpMM().to(self.device)

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.act = None if activation is None else get_activation(activation)

        if residual:
            self.residual = nn.Linear(in_features, out_features * nhead).to(self.device)
        else:
            self.register_buffer("residual", None)
        self.reset_parameters()
        self.alpha = alpha


    def reset_parameters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

        reset(self.a_l) #给这些权重矩阵赋值，均匀分布
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W)
        reset(self.W_e)
        reset(self.edge_emb)

    def forward(self, graph, x, res_attn=None):
        x = self.feat_drop(x)#首先对输入特征做dropout
        h = torch.matmul(x, self.W.to(self.device)).view(-1, self.nhead, self.out_features).to(self.device)#device不一样，有在cpu有在gpu的
        h[torch.isnan(h)] = 0.0 #判断输入的张量是否为0，为空赋值为0
        e = torch.matmul(self.edge_emb.to(self.device), self.W_e.to(self.device)).view(-1, self.nhead, self.edge_feats).to(self.device) #对边计算

        row, col = graph.edge_index #边的索引
        tp = graph.edge_type  #边的类型
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[row]#行权重
        h_r = (self.a_r * h).sum(dim=-1)[col]#列权重
        h_e = (self.a_e * e).sum(dim=-1)[tp]#边类型的权重
        edge_attention = self.leakyrelu(h_l + h_r + h_e) # (num of edge，head)
        # edge_attention: E * H

        edge_attention = edge_softmax(graph, edge_attention)
        edge_attention = self.dropout(edge_attention)
        if res_attn is not None:
            edge_attention = edge_attention * (1 - self.alpha) + res_attn * self.alpha
        out = self.mhspmm(graph, edge_attention, h)#多头注意力，h是wx
        # out的维度是输入节点个数和输出维度
        if self.residual:
            res = self.residual(x)
            out += res
        if self.act is not None:
            out = self.act(out) #ELU激活函数
        out=out.to(self.device)
        return out, edge_attention.detach() #输出out的维度是输出维度×head

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"
