import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
import os
import numpy as np
from torch.distributions.normal import Normal

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, sz_fc = 512, mrg = 0.1, alpha = 32, N = 5, beta = 48, delta = 0.11, r = 1.0, lam = 0.23, gen_mrg = 1.0, scale = 1.0, ratio_gen = 1, ratio_m = 1):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.delta = delta
        self.scale = scale
        self.r = r
        self.lam = lam
        self.gen_mrg = gen_mrg
        self.ratio_gen = ratio_gen
        self.ratio_m = ratio_m
        self.directions = torch.nn.Parameter(torch.randn(self.N, sz_embed).cuda())
        nn.init.kaiming_normal_(self.directions, mode = 'fan_out')
        norm = NormLayer()
        relu = nn.ReLU (True)
        self.fc = nn.Sequential(nn.Linear(sz_embed, sz_fc), norm, relu)
        nn.init.constant_(self.fc[0].bias.data, 0)
        # 正交矩阵
        nn.init.orthogonal_(self.fc[0].weight.data)

    def get_gen_loss1(self, X, P):
        satisfy_size = X.size(0)
        gen_data = self.directions
        # get random directions
        index = [i for i in range(self.N)]
        np.random.shuffle(index)
        gen_data = gen_data[index]
        gen_data = l2_norm(gen_data)  # ..

        gen_loss = list()
        fc = self.fc
        # fc = l2_norm(fc)
        # P = F.linear(P, fc.t())

        for i in range(self.N):
            gen_data[i] = gen_data[i] * (i + 1) * self.r
        #gen_data = l2_norm(gen_data)  # change point
        for i in range(satisfy_size):
            gen_x = torch.empty(self.N, X.size(1)).cuda()
            for j in range(self.N):
                gen_x[j] = X[i] + gen_data[j]
            #gen_x = l2_norm(gen_x)
            # gen_x = l2_norm(F.linear(gen_x, fc.t()))
            gen_x = fc(gen_x)
            Xi = X[i]
            Xi = Xi.view(1, -1)
            # Xi = F.linear(Xi, fc.t())
            Xi = fc(Xi)
            cos_gx = F.linear(gen_x, Xi)
            g_loss_l = list()
            g_loss_r = list()
            for j in range(self.N):
                if j > 0:
                    cos_gx_max_l = 0
                    for p in range(j):
                        cos_gx_max_l += torch.exp(cos_gx[p])
                    cos_gx_max_l = torch.log(cos_gx_max_l)
                    loss_max_left_part = self.beta * (cos_gx[j] - cos_gx_max_l + j * self.delta)
                    g_loss_l.append(loss_max_left_part)
                if j < self.N - 1:
                    cos_gx_max_r = 0
                    for q in range(j + 1, self.N):
                        cos_gx_max_r += torch.exp(cos_gx[q])
                    cos_gx_max_r = torch.log(cos_gx_max_r)
                    # loss_max_right_part = torch.exp(self.beta * (cos_gx_max_r - cos_gx[j] + self.delta))
                    loss_max_right_part = torch.exp(self.beta * (cos_gx[j + 1] - cos_gx[j] + self.delta))
                    g_loss_r.append(loss_max_right_part)
            g_loss_l = sum(g_loss_l)
            g_loss_l = (1 / self.beta) * torch.log(1 + torch.exp(g_loss_l))
            g_loss_r = sum(g_loss_r)
            g_loss_r = (1 / self.beta) * torch.log(1 + g_loss_r)
            gen_loss.append(g_loss_l)
            gen_loss.append(g_loss_r)

            # gen_loss.append(g_loss / self.N)
            P = P.view(1, -1)
            P = fc(P)
            cos_gp = F.linear(gen_x, P)
            metric_loss = torch.tensor([1.0]).cuda()
            for k in range(self.N):
                metric_loss += torch.exp(self.alpha * (self.mrg - cos_gp[k]))
            loss_ = torch.log(torch.sum(metric_loss)) / self.alpha
            gen_loss.append(loss_)

        gen_loss = sum(gen_loss) / satisfy_size
        return gen_loss
        
    def get_gen_loss2(self, X, P): #传进来的是没有l2norm的原数据
        satisfy_size = X.size(0)
        gen_data = self.directions #得到随机的生成方向
        index = [i for i in range(self.N)]
        np.random.shuffle(index)
        gen_data = gen_data[index] #随机打乱生成方向
        gen_data = l2_norm(gen_data) #l2标准化1
        fc = self.fc #Linear, norm, Relu
        for i in range(self.N):
            gen_data[i] = gen_data[i] * (i + 1) * self.r #按照1-N的强度沿不同的方向产生生成向量
        gen_data = l2_norm(gen_data) #l2标准化2，是否冗余？
        gen_loss_gen = list()
        for m in range(satisfy_size):
            gen_x = torch.empty(self.N, X.size(1)).cuda() #返回一个N行，X.size(1)的tensor，用来存放生成样本
            for j in range(self.N):
                gen_x[j] = X[m] + gen_data[j] #生成样本
            gen_x = fc(gen_x) #将生成样本送入定义的fc层
            Xm = X[m] #原始样本
            Xm = Xm.view(1, -1) #调整size
            Xm = fc(Xm) #将Xi送入定义的fc层
            cos_gx = F.linear(gen_x, Xm) #计算生成样本和原样本的余弦相似度
            g_loss_l_mid = list() #存放Lleft的列表
            g_loss_r_mid = list() #存放Lright的列表
            # gen loss
            for i in range(self.N):
                g_loss_l_inner = list()
                g_loss_r_inner = list()

                if i > 0:
                    for j in range(i):
                        loss_l = torch.exp(-self.scale * (cos_gx[i] - cos_gx[j] + i * self.delta))
                        g_loss_l_inner.append(loss_l)
                    g_loss_l_inner_ = sum(g_loss_l_inner)
                    g_loss_l_mid.append(g_loss_l_inner_)

                if i < self.N - 1:
                    for j in range(i + 1, self.N):
                        loss_r = torch.exp(self.scale * (cos_gx[j] - cos_gx[i] + self.delta))
                        g_loss_r_inner.append(loss_r)
                    g_loss_r_inner_ = sum(g_loss_r_inner)
                    g_loss_r_mid.append(g_loss_r_inner_)
            g_loss_l_mid_ = sum(g_loss_l_mid)
            g_loss_r_mid_ = sum(g_loss_r_mid)
            g_loss_l_mid_ = (1 / self.scale) * torch.log(1 + g_loss_l_mid_)
            g_loss_r_mid_ = (1 / self.scale) * torch.log(1 + g_loss_r_mid_)

            gen_loss_gen.append(g_loss_l_mid_)
            gen_loss_gen.append(g_loss_r_mid_)

            #metric loss
            P = P.view(1, -1) #修改proxy的size
            P = fc(P)   #将proxy送入fc
            cos_gp = F.linear(gen_x, P) #计算生成样本和proxy的余弦相似度
            metric_loss = list()
            for k in range(self.N):
                metric_loss.append(torch.exp(self.alpha * (self.mrg - cos_gp[k])))
        metric_loss_ = sum(metric_loss)
        metric_loss = torch.log(1 + metric_loss_) / 1 #这里要除的是proxy的个数，这里只传入了一个proxy
        gen_loss_gen_ = sum(gen_loss_gen)
        loss = (1 - self.ratio) * metric_loss + self.ratio * gen_loss_gen_
        return loss
        
    def get_gen_loss3(self, X, P): #传进来的是没有l2norm的原数据
        satisfy_size = X.size(0)
        gen_data = self.directions #得到随机的生成方向
        index = [i for i in range(self.N)]
        np.random.shuffle(index)
        gen_data = gen_data[index] #随机打乱生成方向
        gen_data = l2_norm(gen_data) #l2标准化1
        fc = self.fc #Linear, norm, Relu
        for i in range(self.N):
            gen_data[i] = gen_data[i] * (i + 1) * self.r #按照1-N的强度沿不同的方向产生生成向量
        gen_data = l2_norm(gen_data) #l2标准化2，是否冗余？
        gen_loss_gen = list()
        for m in range(satisfy_size):
            gen_x = torch.empty(self.N, X.size(1)).cuda() #返回一个N行，X.size(1)的tensor，用来存放生成样本
            for j in range(self.N):
                gen_x[j] = X[m] + gen_data[j] #生成样本
            gen_x = fc(gen_x) #将生成样本送入定义的fc层
            Xm = X[m] #原始样本
            Xm = Xm.view(1, -1) #调整size
            Xm = fc(Xm) #将Xi送入定义的fc层
            cos_gx = F.linear(gen_x, Xm) #计算生成样本和原样本的余弦相似度
            g_loss_l_mid = list() #存放Lleft的列表
            g_loss_r_mid = list() #存放Lright的列表
            # gen loss
            for i in range(self.N):
                g_loss_l_inner = list()
                g_loss_r_inner = list()

                if i > 0:
                    for j in range(i):
                        loss_l = torch.exp(self.scale * (cos_gx[i] - cos_gx[j] + self.delta))
                        g_loss_l_inner.append(loss_l)
                    g_loss_l_inner_ = sum(g_loss_l_inner)
                    g_loss_l_mid.append(g_loss_l_inner_)

                if i < self.N - 1:
                    for j in range(i + 1, self.N):
                        loss_r = torch.exp(self.scale * (cos_gx[j] - cos_gx[i] + self.delta))
                        g_loss_r_inner.append(loss_r)
                    g_loss_r_inner_ = sum(g_loss_r_inner)
                    g_loss_r_mid.append(g_loss_r_inner_)
            g_loss_l_mid_ = sum(g_loss_l_mid)
            g_loss_r_mid_ = sum(g_loss_r_mid)
            g_loss_l_mid_ = (1 / self.scale) * torch.log(1 + g_loss_l_mid_)
            g_loss_r_mid_ = (1 / self.scale) * torch.log(1 + g_loss_r_mid_)

            gen_loss_gen.append(g_loss_l_mid_)
            gen_loss_gen.append(g_loss_r_mid_)

            #metric loss
            P = P.view(1, -1) #修改proxy的size
            P = fc(P)   #将proxy送入fc
            cos_gp = F.linear(gen_x, P) #计算生成样本和proxy的余弦相似度
            metric_loss = list()
            for k in range(self.N):
                metric_loss.append(torch.exp(self.alpha * (self.mrg - cos_gp[k])))
        metric_loss_ = sum(metric_loss)
        metric_loss = torch.log(1 + metric_loss_) / 1 #这里要除的是proxy的个数，这里只传入了一个proxy
        gen_loss_gen_ = sum(gen_loss_gen) / satisfy_size
        #loss = (1 - self.ratio) * metric_loss + self.ratio * gen_loss_gen_
        loss_m = self.ratio_m * metric_loss
        loss_gen = self.ratio_gen * gen_loss_gen_
        loss = loss_m + loss_gen
        return loss

    def forward(self, X, T):
        P = self.proxies
        X_fc = self.fc(X)
        #X = self.fc(X)
        #P = self.fc(P)
        proxy_size = P.size(0)

        #cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        cos = F.linear(l2_norm(X_fc), l2_norm(P))
        #cos = F.linear(X, P)
        cos_px = F.linear(P, X)
        #cos_px = F.linear(l2_norm(P), l2_norm(X))
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        
        gen_loss = list()
        satisfy_cnt = 0
        for i in range(proxy_size):
            pos_index = T == i #标签为i的样本
            satisfy_index = cos_px[i] >= self.gen_mrg #cos_px(100,150),cos_px[i]为对应标签的proxy与所有样本的相似度，选择满足gen_nrg的
            index = torch.mul(pos_index, satisfy_index) #标签为i的样本且与对应的proxy的相似度满足gen_mrg的标签
            X_satisfy = X[index] #标签为i的且与对应的proxy的相似度满足gen_mrg的样本向量
            if X_satisfy.size(0) != 0: #如果存在这样的样本
                satisfy_cnt += X_satisfy.size(0) #保存用于生成的样本数量

                gen_loss.append(self.get_gen_loss3(X_satisfy, P[i])) #得到生成损失
        if satisfy_cnt != 0:
            gen_loss = sum(gen_loss)
            loss = loss + self.lam * gen_loss
        return loss
        


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss