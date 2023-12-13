import torch.nn as nn
from graph.hgn_net import myGATConv
import torch.nn.functional as F
from cogdl.models import BaseModel
import numpy as np
import torch
from cogdl.data import Graph

class SimpleHGN(BaseModel):
    def __init__(
        self,
        args,
        adj,
        num_etypes=5,
        residual=True
    ):
        super(SimpleHGN, self).__init__()
        device=args.device
        self.g = None
        if self.g is None:
            self.build_g_feat(adj,device) #根据邻接矩阵构建图，图中包含边的索引、边的权值和边的类型
        self.in_dims=args.num_features_per_layers[0]
        self.out_dim=args.num_features_per_layers[-1]
        self.num_hidden=args.num_features_per_layers[1]
        self.edge_dim=args.dim_of_edge
        self.num_layers = args.num_of_layers
        self.heads=args.num_heads_per_layer
        self.feat_drop = args.feat_drop
        self.attn_drop = args.attn_drop
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.gat_layers.append(
            myGATConv(
                self.edge_dim, #边的维度是64维
                num_etypes,  #边的类型是5种，这里我应该是3种
                self.in_dims,  #输入的维度768
                self.num_hidden,  #中间层的维度是64
                self.heads[0],   #第一层的头数,1
                device,
                self.feat_drop,  #输入特征的dropout
                self.attn_drop,   #attention的dropout
                False,
                self.activation,
            )
        )

        # hidden layers
        for l in range(1, self.num_layers):  # noqa E741
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                myGATConv(
                    self.edge_dim,
                    num_etypes,
                    self.num_hidden*self.heads[l-1],
                    self.num_hidden,
                    self.heads[l],
                    device,
                    self.feat_drop,
                    self.attn_drop,
                    residual,
                    self.activation,
                )
            )

        # output projection no activation
        self.gat_layers.append(
            myGATConv(
                self.edge_dim,
                num_etypes,
                self.num_hidden*self.heads[-2],
                self.out_dim,
                self.heads[-1],
                device,
                self.feat_drop,
                self.attn_drop,
                residual,
                None,
            )
        )
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))

    def build_g_feat(self, A,device):
        edge2type = {}
        edges = []
        weights = []
        for k, mat in enumerate(A):
            edges.append(mat[0].cpu().numpy()) #mat0是邻接矩阵
            weights.append(mat[1].cpu().numpy()) #weight是权值
            for u, v in zip(*edges[-1]):
                if((u,v) in edge2type): continue
                edge2type[(u, v)] = k
        edges = np.concatenate(edges, axis=1) #把所有的边整合到一个矩阵中
        weights = np.concatenate(weights)
        edges = torch.tensor(edges).to(device)
        weights = torch.tensor(weights).to(device)

        g = Graph(edge_index=edges, edge_weight=weights)#得到一个大图
        g = g.to(device)
        e_feat = []
        for u, v in zip(*g.edge_index):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u, v)])

        e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
        g.edge_type = e_feat
        self.g = g #总共g有三个属性，分别是边的下标，边的权值，边的类型。都根据邻接矩阵得到

    def forward(self, h):
        res_attn = None
        for l in range(self.num_layers):  #这里总共两层 noqa E741
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = h.flatten(1)#应该是针对batch数据，把它按照第二维展开
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits