# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:43:12 2020

@author: zhaog
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABCNN(nn.Module):
    
    def __init__(self, embeddings, num_layer=1, linear_size=300, max_length=50, device="gpu"):
        super(ABCNN, self).__init__()
        self.device = device
        print("embeddings.shape:",embeddings.shape)
        self.embeds_dim = embeddings.shape[1]
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        """
               nn.Parameter(torch.from_numpy(embeddings))
               类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
               并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
               所以在参数优化的时候可以进行优化的
               所以经过类型转换这个self.embed.weight变成了模型的一部分，成为了模型中根据训练可以改动的参数了
               使用nn.Parameter这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
               """
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.embed.float()
        self.embed.weight.requires_grad = True
        self.embed.to(device)
        self.linear_size = linear_size
        self.num_layer = num_layer
        self.conv = nn.ModuleList([Wide_Conv(seq_len=max_length, embeds_size=embeddings.shape[1], device=device) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(
            nn.Linear(self.embeds_dim*(1+self.num_layer)*2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_size, 2),
        )

    def forward(self, q1, q2):
        mask1, mask2 = q1.eq(0), q2.eq(0)
        res = [[], []]
        q1_encode = self.embed(q1)
        q2_encode = self.embed(q2)
        # eg: s1 => res[0]
        # (batch_size, seq_len, dim) => (batch_size, dim)
        # if num_layer == 0
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode
        # batch_size * (dim*(1+num_layer)*2) => batch_size * linear_size
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        sim = self.fc(x)
        probabilities = nn.functional.softmax(sim, dim=-1)
        return sim, probabilities


class Wide_Conv(nn.Module):
    def __init__(self, seq_len, embeds_size, device="gpu"):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        """
        nn.Parameter(torch.randn((seq_len, embeds_size)))
        类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的
        所以经过类型转换这个self.W变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        使用nn.Parameter这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
        """
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.W.to(device)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()
        
    def forward(self, sent1, sent2, mask1, mask2):
        '''
        sent1, sent2: batch_size * seq_len * dim
        '''
        # sent1, sent2 = sent1.transpose(0, 1), sent2.transpose(0, 1)
        # => A: batch_size * seq_len * seq_len
        # 计算attention矩阵  A就是attention矩阵  根据两个句子的feature map生成一个Attention matrix A
        A = match_score(sent1, sent2, mask1, mask2)   #根据两个句子的feature map生成一个Attention matrix A
        # attn_feature_map1: batch_size * seq_len * dim
        print("A.shape:",A.shape)
        print("self.W.shape:",self.W.shape)
        """
        基本的想法是在卷积之前，根据两个句子的feature map生成一个Attention matrix A，
        之后将矩阵A与参数W进行矩阵乘法，构造一个与原始feature map大小一致的新attention feature map作为卷积输入的另一个通道，
        之后再进行卷积池化等操作，希望attention feature map能够在卷积操作时起到有关注点的抽取特征，
        """
        attn_feature_map1 = A.matmul(self.W)  #计算句子的attention特征（模型参数矩阵 * attention矩阵）
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)  #第二个句子的attention矩阵也要计算
        # x1: batch_size * 2 *seq_len * dim
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1) #unsqueeze(1) 数据第二维度进行扩充  torch.cat(,1)横着拼接tensor
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)  #squeeze(1) 去除第二维度
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2

# 计算attention矩阵  根据两个句子的feature map生成一个Attention matrix A
def match_score(s1, s2, mask1, mask2):
    '''根据两个句子的feature map生成一个Attention matrix A
    s1, s2:  batch_size * seq_len  * dim
    '''
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)

def attention_avg_pooling(sent1, sent2, mask1, mask2):
    # A: batch_size * seq_len * seq_len
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2