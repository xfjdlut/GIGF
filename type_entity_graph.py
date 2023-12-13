import numpy as np
import json

import torch
from scipy.sparse import diags, csr, csr_matrix
from sparse import eye

def json_read(path):
    with open(path,'r') as f:
        data=json.loads(f.read())
    return data


def type_entity_dic(type,typedic,entity,entitydic):
    type_dict={}
    entity_dict={}
    with open(type,'r') as ft:
        for idx,line in enumerate(ft.readlines()):
            type_dict[line.replace('\n','')]=idx
            type_len=type_len+1
    with open(entity,'r') as fe:
        for idx,line in enumerate(fe.readlines()):
            entity_dict[line.replace('\n','')]=idx
            entity_len=entity_len+1
    with open(typedic, 'w',encoding='utf-8') as fp:
        fp.write(json.dumps(type_dict,ensure_ascii=False))
    with open(entitydic, 'w',encoding='utf-8') as fp:
        fp.write(json.dumps(entity_dict,ensure_ascii=False))

#构建type和entity的异构图
class TypeEntityGraph():
    def __init__(self,typedic,entitydic,train,dev,kcbalance):
        self.type_dic_path=typedic
        self.entity_dic_path=entitydic
        self.train_path=train
        self.dev_path=dev
        self.kcbalance=kcbalance

    # 创建type和entity的连接图
    def fusion_graph(self):
        graph_all=self.create_type_entity_graph(self)
        type_len=len(graph_all[0])
        entity_len=len(graph_all[2])
        fusion_len=type_len+entity_len
        fusion_graph=np.zeros([fusion_len, fusion_len],dtype=int)
        g={}
        g['adj']=[]
        adj_type_i=[]
        adj_type_j=[]
        adj_type_v=[]
        for i in range(len(graph_all[0])):
            for j in range(len(graph_all[0][i])):
                fusion_graph[i][j]=graph_all[0][i][j]
                if(graph_all[0][i][j]):
                    adj_type_i.append(i)
                    adj_type_j.append(j)
                    adj_type_v.append(1)
        adj_type=torch.stack([torch.tensor(adj_type_i),torch.tensor(adj_type_j)],0)
        adjtype=(adj_type,torch.tensor(adj_type_v))
        g['adj'].append(adjtype)
        adj_entity_i=[]
        adj_entity_j=[]
        adj_entity_v=[]

        # 这个地方是entity
        for i in range(len(graph_all[2])):
            for j in range(len(graph_all[2][i])):
                fusion_graph[i+type_len][j+type_len]=graph_all[2][i][j]
                if(graph_all[2][i][j]):
                    adj_entity_i.append(i+type_len)
                    adj_entity_j.append(j+type_len)
                    adj_entity_v.append(1)
        adj_entity = torch.stack([torch.tensor(adj_entity_i), torch.tensor(adj_entity_j)], 0)
        adjentity=(adj_entity,torch.tensor(adj_entity_v))
        g['adj'].append(adjentity)
        adj_type_entity_i = []
        adj_type_entity_j = []
        adj_type_entity_v = []

        # 这个地方是type_entity 和 entity_type
        for i in range(len(graph_all[4])):
            for j in range(len(graph_all[4][i])):
                fusion_graph[i][j+type_len]=graph_all[4][i][j]
                if (graph_all[4][i][j]):
                    adj_type_entity_i.append(i)
                    adj_type_entity_j.append(j+type_len)
                    adj_type_entity_v.append(1)
        adj_type_entity = torch.stack([torch.tensor(adj_type_entity_i),torch.tensor(adj_type_entity_j)],0)
        adjtypeentity = (adj_type_entity, torch.tensor(adj_type_entity_v))
        g['adj'].append(adjtypeentity)

        adj_entity_type =torch.stack([torch.tensor(adj_type_entity_j),torch.tensor(adj_type_entity_i)],0)
        adjentitytype = (adj_entity_type, torch.tensor(adj_type_entity_v))
        g['adj'].append(adjentitytype)

        # 创建自边
        adj_self_edge=torch.arange(0,fusion_len)
        adj_self_value=torch.ones(fusion_len)
        adj_self=torch.stack([adj_self_edge,adj_self_edge],0)
        adjself=(adj_self,adj_self_value)
        g['adj'].append(adjself)
        return g, graph_all[-1],type_len,entity_len


    @staticmethod
    def create_type_entity_graph(self):
        self.type_dic=json_read(self.type_dic_path)
        self.entity_dic=json_read(self.entity_dic_path)
        type_graph,type_adj=self._create_type_graph(self,self.type_dic,self.train_path,self.dev_path)   #构建type图
        entity_graph,entity_adj=self._create_entity_graph(self, self.entity_dic, self.train_path, self.dev_path)   #构建entity图

        type_len =list(self.type_dic.values())[-1]+1
        entity_len = list(self.entity_dic.values())[-1]+1
        # type和entity相邻的图和邻接矩阵
        type_entity_graph = np.zeros([type_len, entity_len],dtype=int)
        type_entity_adj = np.zeros([type_len, entity_len],dtype=int)

        id_before = -1
        with open(self.train_path, 'r',encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                type_path = sample['action_path']
                entity_path=sample['topic_path']
                for j in range(len(type_path)):
                    entity=entity_path[j]
                    idx = self.type_dic[type_path[j]]
                    jdx = self.entity_dic[entity]
                    type_entity_graph[idx][jdx] = 1
                    type_entity_adj[idx][jdx] = type_entity_adj[idx][jdx] + 1
                id_before=id_now
        id_before = -1
        with open(self.dev_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                type_path = sample['action_path']
                entity_path = sample['topic_path']
                for j in range(len(type_path)):
                    entity = entity_path[j]
                    idx = self.type_dic[type_path[j]]
                    jdx = self.entity_dic[entity]
                    type_entity_graph[idx][jdx] = 1
                    type_entity_adj[idx][jdx] = type_entity_adj[idx][jdx] + 1
                id_before = id_now

        return [type_graph, type_adj, entity_graph, entity_adj, type_entity_graph, type_entity_adj]

    # 这两个图都没有计算最后的再见和NULL，但是过程中的NULL没有去掉
    @staticmethod
    def _create_type_graph(self,type_dic,train_path,dev_path):
        type_len = list(type_dic.values())[-1] + 1
        # type的转换图和邻接矩阵的值
        type_graph = np.zeros([type_len, type_len], dtype=int)
        type_adj = np.zeros([type_len, type_len], dtype=int)
        # 得到图和邻接矩阵，邻接矩阵是带权重的图
        id_before = -1
        with open(train_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                type_path = sample['action_path']
                for j in range(1, len(type_path) - 1):  #不算最后的再见
                    idx = type_dic[type_path[j]]
                    jdx = type_dic[type_path[j - 1]]
                    type_graph[idx][jdx] = 1
                    type_adj[idx][jdx] = type_adj[idx][jdx] + 1
                id_before = id_now

        id_before = -1
        with open(dev_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                type_path = sample['action_path']
                for j in range(1, len(type_path) - 1):
                    idx = type_dic[type_path[j]]
                    jdx = type_dic[type_path[j - 1]]
                    type_graph[idx][jdx] = 1
                    type_adj[idx][jdx] = type_adj[idx][jdx] + 1
                id_before = id_now
        return type_graph, type_adj

    # 融合知识图和共现图
    @staticmethod
    def _create_entity_graph(self, entity_dic, train_path, dev_path):
        entity_cgraph,entity_cadj=self._create_entity_cgraph(self,entity_dic,train_path,dev_path)
        entity_kgraph,entity_kadj=self._create_entity_kgraph(self,entity_dic,train_path,dev_path)
        entity_cadj=self._normalize_adj(entity_cadj)
        entity_kadj=self._normalize_adj(entity_kadj)
        entity_adj=entity_kadj
        entity_graph=entity_cgraph
        for i in range(len(entity_adj)):
            for j in range(len(entity_adj[i])):
                entity_adj[i][j]=self.kcbalance*entity_cadj[i][j]+(1-self.kcbalance)*entity_kadj[i][j]
                if entity_adj[i][j]:
                    entity_graph[i][j]=1
                else:
                    entity_graph[i][j]=0
        return entity_graph,entity_adj

    # entity的知识图构建knowledge
    @staticmethod
    def _create_entity_kgraph(self, entity_dic, train_path, dev_path):
        entity_len = list(entity_dic.values())[-1] + 1
        entity = list(entity_dic.keys())
        # entity的转换图和邻接矩阵的值
        entity_kgraph = np.zeros([entity_len, entity_len], dtype=int)
        entity_kgadj = np.zeros([entity_len, entity_len], dtype=int)
        # 训练集知识构图
        id_before = -1
        with open(train_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                entity_kg = sample['knowledge']
                for s, p, o in entity_kg:
                    if s in entity and o in entity:
                        idx = entity_dic[s]
                        jdx = entity_dic[o]
                        entity_kgraph[idx][jdx] = 1
                        entity_kgraph[jdx][idx] = 1
                        entity_kgadj[idx][jdx] = entity_kgadj[idx][jdx] + 1
                        entity_kgadj[jdx][idx] = entity_kgadj[jdx][idx] + 1
                id_before = id_now
        # 验证集
        id_before = -1
        with open(dev_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                entity_kg = sample['knowledge']
                for s, p, o in entity_kg:
                    if s in entity and o in entity:
                        idx = entity_dic[s]
                        jdx = entity_dic[o]
                        entity_kgraph[idx][jdx] = 1
                        entity_kgraph[jdx][idx] = 1
                        entity_kgadj[idx][jdx] = entity_kgadj[idx][jdx] + 1
                        entity_kgadj[jdx][idx] = entity_kgadj[jdx][idx] + 1
                id_before = id_now
        return entity_kgraph, entity_kgadj

    # entity的exchange图构建
    @staticmethod
    def _create_entity_cgraph(self,entity_dic,train_path,dev_path):
        entity_len = list(entity_dic.values())[-1] + 1
        # entity的转换图和邻接矩阵的值
        entity_graph = np.zeros([entity_len, entity_len], dtype=int)
        entity_adj = np.zeros([entity_len, entity_len], dtype=int)

        id_before = -1
        with open(train_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                entity_path = sample['topic_path']
                for j in range(1, len(entity_path) - 1):  # 不算最后的NULL
                    idx = entity_dic[entity_path[j]]
                    jdx = entity_dic[entity_path[j - 1]]
                    entity_graph[idx][jdx] = 1
                    entity_adj[idx][jdx] = entity_adj[idx][jdx] + 1
                id_before = id_now

        id_before = -1
        with open(dev_path, 'r', encoding='utf-8') as fp:
            for line in fp:
                sample = json.loads(line.strip())
                id_now = sample['id']
                if (id_now == id_before):
                    continue
                entity_path = sample['topic_path']
                for j in range(1, len(entity_path) - 1):
                    idx = entity_dic[entity_path[j]]
                    jdx = entity_dic[entity_path[j - 1]]
                    entity_graph[idx][jdx] = 1
                    entity_adj[idx][jdx] = entity_adj[idx][jdx] + 1
                id_before = id_now
        return entity_graph, entity_adj

    @staticmethod
    def _normalize_adj(mx):
        if type(mx) is not csr.csr_matrix:
            row,col=np.nonzero(mx)
            values=mx[row,col]
            csr_mx=csr_matrix((values,(row,col)),shape=(len(mx),len(mx[0])))
        if csr_mx[0, 0] == 0:  #判断是否有自边，没有的话，就加上
            csr_mx = csr_mx + eye(csr_mx.shape[0]).tocsr()
        rowsum = np.array(csr_mx.sum(1))
        r_inv = np.power(rowsum, -1 / 2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = diags(r_inv)
        csr_mx = r_mat_inv.dot(csr_mx)
        csr_mx = csr_mx.dot(r_mat_inv)
        nor_mx=csr_mx.toarray()
        return nor_mx
