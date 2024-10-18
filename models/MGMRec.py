# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from models.GAT import GAT
from utils.utils import normalize_adjacency


class My(GeneralRecommender):
    def __init__(self, config, dataset):
        super(My, self).__init__(config, dataset)  # 调用父类的构造函数，传入配置和数据集

        self.embedding_dim = config['embedding_size']  # 设置嵌入向量的维度大小
        self.feat_embed_dim = config['feat_embed_dim']  # 设置特征嵌入的维度大小
        self.cf_model = config['cf_model']  # 协同过滤模型的类型
        self.n_mm_layer = config['n_mm_layers']  # 多模态图嵌入的层数
        self.n_ui_layers = config['n_ui_layers']  # 用户-项目图嵌入的层数
        self.reg_weight = config['reg_weight']  # 正则化项的权重
        self.tau = 0.2  # Gumbel softmax的温度参数
        self.softmax = nn.Softmax(dim=-1)
        self.n_nodes = self.n_users + self.n_items  # 总节点数（用户数 + 项目数）
        self.meta_weight=config['meta_weight']
        self.n_meta_layer=config['n_meta_layer']
        # 加载数据集信息
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)  # 获取用户-项目交互矩阵，并转换为COO格式
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix,
                                                      torch.Size((self.n_users, self.n_items)))  # 将交互矩阵转换为稀疏张量
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()  # 计算并归一化邻接矩阵
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)  # 计算交互次数的倒数并转换为张量

        # 初始化用户和项目ID的嵌入向量
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)  # 初始化用户嵌入矩阵
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)  # 初始化项目嵌入矩阵
        nn.init.xavier_uniform_(self.user_embedding.weight)  # 使用Xavier均匀分布初始化用户嵌入矩阵
        nn.init.xavier_uniform_(self.item_id_embedding.weight)  # 使用Xavier均匀分布初始化项目嵌入矩阵

        self.drop = nn.Dropout(p=0.5)  # 初始化Dropout层，用于防止过拟合
        self.build_metapath()
        # 加载项目的多模态特征并定义超图嵌入向量
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)  # 加载预训练的图像特征嵌入矩阵，并冻结参数
            self.item_image_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))  # 初始化图像特征转换矩阵
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)  # 加载预训练的文本特征嵌入矩阵，并冻结参数
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))  # 初始化文本特征转换矩阵

    def build_metapath(self):
        A_dense = torch.tensor(self.interaction_matrix.toarray(), dtype=torch.float32)
        A_T = A_dense.T
        self.user_item_user = normalize_adjacency(torch.matmul(A_dense, A_T)).to_sparse().to(self.device)

        A_T = A_dense.T
        self.item_user_item = normalize_adjacency(torch.matmul(A_T, A_dense)).to_sparse().to(self.device)

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row  # 获取稀疏矩阵的行索引
        col = matrix.col  # 获取稀疏矩阵的列索引
        i = torch.LongTensor(np.array([row, col]))  # 构建稀疏张量的索引矩阵
        data = torch.FloatTensor(matrix.data)  # 获取稀疏矩阵的非零元素
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)  # 创建稀疏张量并转换为特定设备上使用

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)  # 创建一个稀疏的邻接矩阵
        inter_M = self.interaction_matrix  # 获取用户-项目交互矩阵
        inter_M_t = self.interaction_matrix.transpose()  # 交互矩阵的转置
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))  # 将用户-项目交互矩阵转换为字典形式
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))  # 更新字典，加入转置矩阵的信息
        A._update(data_dict)  # 更新邻接矩阵的数据

        sumArr = (A > 0).sum(axis=1)  # 计算每个节点的度
        diag = np.array(sumArr.flatten())[0] + 1e-7  # 获取度的对角矩阵，并加上一个小数以避免除零错误
        diag = np.power(diag, -0.5)  # 计算度的-0.5次幂
        D = sp.diags(diag)  # 创建度的对角矩阵
        L = D * A * D  # 计算归一化后的拉普拉斯矩阵
        L = sp.coo_matrix(L)  # 转换为COO格式的稀疏矩阵
        return sumArr, self.scipy_matrix_to_sparse_tenser(L,
                                                          torch.Size((self.n_nodes, self.n_nodes)))  # 返回度矩阵和归一化后的邻接矩阵

    # 协同图嵌入
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)  # 对用户和项目嵌入进行拼接
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight),
                                       dim=0)  # 获取初始的用户和项目嵌入
            cge_embs = [ego_embeddings]  # 初始化嵌入列表
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)  # 进行多层用户-项目图嵌入的传播计算
                cge_embs += [ego_embeddings]  # 将计算结果添加到嵌入列表中
            cge_embs = torch.stack(cge_embs, dim=1)  # 对嵌入进行堆叠
            cge_embs = cge_embs.mean(dim=1, keepdim=False)  # 取嵌入的平均值
        if self.cf_model == 'gat':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight),
                                       dim=0)  # 获取初始的用户和项目嵌入
            cge_embs = [ego_embeddings]  # 初始化嵌入列表
            for _ in range(5):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)  # 进行多层用户-项目图嵌入的传播计算
                cge_embs += [ego_embeddings]  # 将计算结果添加到嵌入列表中
            cge_embs = torch.stack(cge_embs, dim=1)  # 对嵌入进行堆叠
            cge_embs = cge_embs.mean(dim=1,
                                     keepdim=False)  # 取嵌入的平均值  # cge_embs = self.GAT_model(ego_embeddings, self.norm_adj)
        return cge_embs  # 返回协同图嵌入结果

    # 模态图嵌入
    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)  # 计算图像模态的项目嵌入
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)  # 计算文本模态的项目嵌入
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]  # 根据项目嵌入计算用户嵌入，并乘以交互次数的倒数
        mge_feats = torch.concat([user_feats, item_feats], dim=0)  # 将用户和项目嵌入进行拼接
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)  # 进行多层模态图嵌入的传播计算
        return mge_feats  # 返回模态图嵌入结果

    def meta(self, u_embs_u, i_embs_i):
        u_mate_embs = [u_embs_u]  # 初始化嵌入列表
        i_mate_embs = [i_embs_i]
        for _ in range(self.n_meta_layer):
            u_embs_u = torch.sparse.mm(self.user_item_user, u_embs_u)  # 进行多层用户-项目图嵌入的传播计算
            u_mate_embs += [u_embs_u]  # 将计算结果添加到嵌入列表中
            i_embs_i = torch.sparse.mm(self.item_user_item, i_embs_i)  # 进行多层用户-项目图嵌入的传播计算
            i_mate_embs += [i_embs_i]  # 将计算结果添加到嵌入列表中
        u_embs = torch.stack(u_mate_embs, dim=1)
        u_embs = u_embs.mean(dim=1, keepdim=False)
        i_embs = torch.stack(i_mate_embs, dim=1)
        i_embs = i_embs.mean(dim=1, keepdim=False)
        return u_embs, i_embs

    def forward(self):
        cge_embs = self.cge()  # 计算协同图嵌入
        u_embs_u, i_embs_i = torch.split(cge_embs, [self.n_users, self.n_items], dim=0)
        u_meta_embs, i_meta_embs = self.meta(u_embs_u, i_embs_i)
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: 模态图嵌入
            v_feats = F.normalize(self.mge('v'))  # 计算图像模态的嵌入
            t_feats = F.normalize(self.mge('t'))  # 计算文本模态的嵌入
            # 本地嵌入 = 协同相关嵌入 + 模态相关嵌入
            mge_embs = v_feats + t_feats  # 对模态图嵌入进行归一化，并相加
            lge_embs = cge_embs + mge_embs  # 本地嵌入为协同图嵌入与模态图嵌入之和
            all_embs = lge_embs
        else:
            all_embs = cge_embs  # 如果没有模态特征，则仅使用协同图嵌入作为最终嵌入
        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)  # 将所有嵌入分割为用户嵌入和项目嵌入
        u_embs, i_embs = u_embs +self.meta_weight * u_meta_embs ,  i_embs+ self.meta_weight * i_meta_embs
        return u_embs, i_embs, mge_embs, v_feats, t_feats

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):

        ua_embeddings, ia_embeddings, mge_embs, v_feats, t_feats = self.forward()
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = batch_bpr_loss + self.reg_weight * batch_reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, mge_embs, v_feats, t_feats = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores
