import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim).cuda()
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.linear3 = nn.Linear(self.hidden_dim, self.output_dim).cuda()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(self.dropout(x))
        x = self.linear2(x)
        x = self.relu(self.dropout(x))
        x = self.linear3(x)
        x = self.relu(self.dropout(x))
        return x


def feature_matching(teacher_feature, student_feature):
    ''' calculate matching score between teacher model and student model  '''
    _, c1, h1, w1 = teacher_feature.shape
    _, c2, h2, w2 = student_feature.shape
    # print(teacher_feature.shape,student_feature.shape)
    if h1 + w1 == h2 + w2:
        if c1 == c2:
            dif = teacher_feature - student_feature
            score = torch.sqrt(dif.pow(2)).mean()
            return score
        else:
            teacher_feature = teacher_feature.reshape(-1, h1 * w1, c1)
            student_feature = student_feature.reshape(-1, h2 * w2, c2)
            student_feature = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_feature)
            dif = teacher_feature - student_feature
            score = torch.sqrt(dif.pow(2)).mean()
            return score

    else:
        return None


def batch_select(scores, pairs):
    ''' select the position with best match. i.e. the minimum of matching score '''
    # print(len(scores))
    min_score = scores[0]
    min_score_index = 0
    for i in range(len(scores)):
        if scores[i] <= min_score:
            min_score_index = i

    i, j = pairs[min_score_index]

    return i, j


def batch_matching(teacher_block, student_block):
    ''' Automatically adjust distill stage. Note that there could be a one-to-many situation '''
    scores = []
    pairs = []
    for i in range(len(teacher_block)):
        _, c1, h1, w1 = teacher_block[i].shape
        for j in range(len(student_block)):
            _, c2, h2, w2 = student_block[j].shape
            if h1 + w1 == h2 + w2:
                score = feature_matching(teacher_block[i], student_block[j])
                # print(score.item())
                scores.append(score.item())
                pairs.append([i, j])
            else:
                continue

    # print('len-scores---',len(scores))
    B_t, B_s = batch_select(scores, pairs)
    B_s += 1
    B_t += 1
    # print('batch matching---:',B_t,B_s)

    pairs = []
    for i in range(B_t ):
        _, c1, h1, w1 = teacher_block[i].shape
        # print(h1,w1)
        for j in range(B_s ):
            _, c2, h2, w2 = student_block[j].shape
            # print(h2,w2)
            if h1 + w1 == h2 + w2:
                pairs.append([i, j])
            else:
                continue
    # print('---',pairs)
    return B_t, B_s,pairs

def batch_load(teacher_block,student_block):
    B_t = len(teacher_block)
    B_s = len(student_block)
    pairs = []
    for i in range(B_t):
        _, c1, h1, w1 = teacher_block[i].shape
        for j in range(B_s):
            _, c2, h2, w2 = student_block[j].shape
            if h1 + w1 == h2 + w2:
                pairs.append([i, j])
            else:
                continue
    return B_t,B_s,pairs



class Cross_layer_weight(nn.Module):
    ''' calculate the weight of each match pair of batch'''

    def __init__(self,if_matching):
        super(Cross_layer_weight, self).__init__()
        self.cosin = nn.CosineSimilarity()
        self.if_matching = if_matching

    def forward(self, T_block, S_block):
        weight_layer = []
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if self.if_matching:
            _, _, pairs = batch_matching(T_block, S_block)
        else:
            _,_,pairs = batch_load(T_block, S_block)
        # print(self.if_matching)
        # print('paris',pairs)
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        for i in range(len(pairs)):
            # print(i)
            T_feature = T_block[pairs[i][0]]
            S_feature = S_block[pairs[i][1]]
            _, c1, h1, w1 = T_feature.shape
            _, c2, h2, w2 = S_feature.shape
            assert h1 + w1 == h2 + w2
            if c1 == c2:
                dif = self.cosin(T_feature, S_feature).flatten().mean()
                weight_layer.append(dif)
            else:
                T_feature = T_feature.reshape(-1,h1*w1,c1)
                S_feature = S_feature.reshape(-1,h2*w2,c2)
                S_feature = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(S_feature)
                dif = self.cosin(T_feature, S_feature).flatten().mean()
                weight_layer.append(dif)
        B = torch.stack(weight_layer, dim=-1)
        weight = F.softmax(B, dim=-1)

        # print('len_cross_layer_weight',len(weight))
        return weight

# torch.random.seed()
# a = torch.randn(24,3,192,256)*100
# aa = torch.randn(24,3,192,256)*100
# b = torch.ones(24,50,96,128)*100
# bb = torch.randn(24,50,96,128)*100
# c = torch.randn(24,100,48,64)*100
# cc = torch.ones(24,100,48,64)*100
# d = torch.randn(24,100,48,64)*100
# e = torch.randn(24,100,48,64)*100
# teacher = [a,b,c,d,e]
# student = [aa,bb,cc]
#
# scores,pairs = batch_matching(teacher,student)
# print(scores,pairs)

class Cross_layer_attention(nn.Module):
    ''' Attention for the eaqual size of teacher feature and student feature '''

    def __init__(self, in_channel, out_channel):
        super(Cross_layer_attention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1x1_init = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=(1, 1), padding=0)
        self.conv3x3 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(3, 3), padding=1)
        self.conv1x1 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(1, 1), padding=0)
        self.tran = nn.Sequential(
            self.conv1x1_init,
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            self.conv3x3,
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
            self.conv1x1,
        ).cuda()

    def forward(self, T_feature, S_feature):
        b1, c1, h1, w1 = T_feature.shape
        b2, c2, h2, w2 = S_feature.shape
        assert b1 == b2
        if c1 == c2:
            # print("!!!!",self.tran(T_feature).shape)
            T_feature = self.tran(T_feature).reshape(b1, -1, c1)
            # print("!!!!",T_feature.shape)

            S_feature = self.conv1x1(S_feature).reshape(b2, -1, c2)
        else:
            T_feature = T_feature.reshape(b1, -1, c1)
            S_feature = S_feature.reshape(b2, -1, c2)
            S_feature = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(S_feature)

        # attention weight
        # print(T_feature.shape,S_feature.shape)
        product = torch.bmm(T_feature.permute(0,2,1), S_feature)
        weight = F.softmax(product, dim=-1)

        return weight

# a = torch.ones(24,3,192,256)*100
# aa = torch.ones(24,3,192,256)*100
# b = torch.ones(24,50,96,128)*100
# bb = torch.ones(24,50,96,128)*100
# c = torch.ones(24,100,48,64)*100
# cc = torch.ones(24,100,48,64)*100
# d = torch.zeros(24,100,48,64)*100
# e = torch.zeros(24,100,48,64)*100
# teacher = [a,b,c,d,e]
# student = [aa,bb,cc]
#
# Clw = Cross_layer_attention()
# w = Clw(teacher,student)
# print(w)
