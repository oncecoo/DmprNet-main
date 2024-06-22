import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from distill_loss.cross_layer_attention import *


def add_weight(T_block, S_block,if_matching):
    '''For each matching feature pair, calculate the pair weight
        return => list   which is the product of weight and  for adjust batch(equal lenghth)
    '''
    # print('length of T_b,S_b',len(T_block),len(S_block))
    CL_weight = Cross_layer_weight(if_matching=if_matching)
    block_weight = CL_weight(T_block, S_block).cpu()
    
    if if_matching:
        # print('###########################batch_matching')
        B_t, B_s, _ = batch_matching(T_block, S_block)

    else:
        B_t,B_s ,_= batch_load(T_block, S_block)
    weights = []
    # print('###############################################')
    # print(B_t,B_s)
    for i in range(B_t):
        T_feature = T_block[i].cuda()
        # print(T_feature.shape)
        b1, c1, h1, w1 = T_feature.shape
        for j in range(B_s):
            S_feature = S_block[j].cuda()
            b2, c2, h2, w2 = S_feature.shape
            # print(S_feature.shape)
            if h1+w1 == h2+w2:
                CL_atten = Cross_layer_attention(c1, c2)
                weight = CL_atten(T_feature,S_feature)
                weights.append(weight)
            else:
                continue

    block_weight = block_weight.detach().numpy().tolist()
    # print(len(weights),len(block_weight))

    # assert len(weights) == len(block_weight)
    W = []
    for i in range(len(weights)):
        W.append(weights[i] * block_weight[i])
    return W


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
# a = add_weight(teacher,student,False)
# print(len(a),a[0].shape,a[1].shape)

'''
最终得到的是一个列表，
'''
class FMDLoss(nn.Module):
    def __init__(self):
        super(FMDLoss, self).__init__()

        self.MSELoss = nn.MSELoss(size_average=True, reduce=True,reduction='none')

    def forward(self, teacher_block, student_block):
        assert len(teacher_block) == len(student_block) #B_s == B_t
        losses = []
        loss = []
        alpha,lamada,beta = 0.004,0.002,0.001
        for stage in range(len(teacher_block)):
            if stage == 0:
                # print('stage:',stage)
                _,_,pairs = batch_matching(teacher_block[0],student_block[0])
                weight = add_weight(teacher_block[0], student_block[0], if_matching=True)

                for i in range(len(pairs)):

                    b1,c1,h1,w1 =  teacher_block[0][pairs[i][0]].shape
                    b2,c2,h2,w2 = student_block[0][pairs[i][0]].shape
                    if c1 == c2:
                        teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, c1, h1 * w1)
                        student_f = student_block[0][pairs[i][1]].reshape(b2, c2, h2 * w2)
                    else:
                        teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, -1, c1)
                        student_f = student_block[0][pairs[i][1]].reshape(b2, -1, c2)
                        student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)


                    loss.append(self.MSELoss(teacher_f,student_f))
                product = list(map(lambda x, y: x * y, weight,loss))
                count = 0.
                for i in range(len(product)):
                    count += product[i].mean()
                count /= len(product)
                losses.append(count)

            if stage==1:
                # print('stage:',stage)

                _,_,pairs = batch_matching(teacher_block[1],student_block[1])
                # print('------------------------------------------------------------------')
                weight_1 = add_weight(teacher_block[1], student_block[1], if_matching=True)
                # print('--------------------------------------------------------------------')
                for i in range(len(pairs)):
                        b1,c1,h1,w1 =  teacher_block[1][pairs[i][0]].shape
                        b2,c2,h2,w2 = student_block[1][pairs[i][1]].shape
                        if c1 == c2:
                            teacher_f = teacher_block[1][pairs[i][0]].reshape(b1, c1, h1 * w1)
                            student_f = student_block[1][pairs[i][1]].reshape(b2, c2, h2 * w2)
                        else:
                            teacher_f = teacher_block[1][pairs[i][0]].reshape(b1, -1, c1)
                            student_f = student_block[1][pairs[i][1]].reshape(b2, -1, c2)
                            student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)

                        loss.append(self.MSELoss(teacher_f,student_f))
                product = list(map(lambda x, y: x * y, weight_1,loss))
                count = 0.
                for i in range(len(product)):
                    count += product[i].mean()
                count /= len(product)

                weight_0 = add_weight(teacher_block[0], student_block[0], if_matching=False)
                _,_,pairs = batch_load(teacher_block[0],student_block[0])
                for i in range(len(pairs)):
                        b1, c1, h1, w1 = teacher_block[0][pairs[i][0]].shape
                        b2, c2, h2, w2 = student_block[0][pairs[i][1]].shape
                        if c1 == c2:
                            teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, c1, h1 * w1)
                            student_f = student_block[0][pairs[i][1]].reshape(b2, c2, h2 * w2)
                        else:
                            teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, -1, c1)
                            student_f = student_block[0][pairs[i][1]].reshape(b2, -1, c2)
                            student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)

                        loss.append(self.MSELoss(teacher_f, student_f))
                product = list(map(lambda x, y: x * y, weight_0, loss))
                count_0 = 0.
                for i in range(len(product)):
                    count_0 += product[i].mean()
                    # print(count_0)
                count_0 /= len(product)
                count += count_0
                losses.append(count)

            if stage == 2:
                # print('stage:',stage)

                _, _, pairs = batch_matching(teacher_block[2], student_block[2])
                weight_2 = add_weight(teacher_block[2], student_block[2], if_matching=True)
                for i in range(len(pairs)):
                        b1, c1, h1, w1 = teacher_block[2][pairs[i][0]].shape
                        b2, c2, h2, w2 = student_block[2][pairs[i][1]].shape
                        if c1 == c2:
                            teacher_f = teacher_block[2][pairs[i][0]].reshape(b1, c1, h1 * w1)
                            student_f = student_block[2][pairs[i][1]].reshape(b2, c2, h2 * w2)
                        else:
                            teacher_f = teacher_block[2][pairs[i][0]].reshape(b1, -1, c1)
                            student_f = student_block[2][pairs[i][1]].reshape(b2, -1, c2)
                            student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)

                        loss.append(self.MSELoss(teacher_f, student_f))
                product = list(map(lambda x, y: x * y, weight_2, loss))
                count = 0.
                for i in range(len(product)):
                    count += product[i].mean()
                count /= len(product)


                weight_0 = add_weight(teacher_block[0], student_block[0], if_matching=False)
                _, _, pairs = batch_load(teacher_block[0], student_block[0])
                for i in range(len(pairs)):

                        b1, c1, h1, w1 = teacher_block[0][pairs[i][0]].shape
                        b2, c2, h2, w2 = student_block[0][pairs[i][1]].shape
                        if c1 == c2:
                            teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, c1, h1 * w1)
                            student_f = student_block[0][pairs[i][1]].reshape(b2, c2, h2 * w2)
                        else:
                            teacher_f = teacher_block[0][pairs[i][0]].reshape(b1, -1, c1)
                            student_f = student_block[0][pairs[i][1]].reshape(b2, -1, c2)
                            student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)

                        loss.append(self.MSELoss(teacher_f, student_f))
                product = list(map(lambda x, y: x * y, weight_0, loss))
                count_0 = 0.
                for i in range(len(product)):
                    count += product[i].mean()
                count_0 /= len(product)
                count += count_0

                weight_1 = add_weight(teacher_block[1], student_block[1], if_matching=False)
                _, _, pairs = batch_load(teacher_block[1], student_block[1])
                for i in range(len(pairs)):
                        b1, c1, h1, w1 = teacher_block[1][pairs[i][0]].shape
                        b2, c2, h2, w2 = student_block[1][pairs[i][1]].shape
                        if c1 == c2:
                            teacher_f = teacher_block[1][pairs[i][0]].reshape(b1, c1, h1 * w1)
                            student_f = student_block[1][pairs[i][1]].reshape(b2, c2, h2 * w2)
                        else:
                            teacher_f = teacher_block[1][pairs[i][0]].reshape(b1, -1, c1)
                            student_f = student_block[1][pairs[i][1]].reshape(b2, -1, c2)
                            student_f = MLP(input_dim=c2, hidden_dim=2 * c2, output_dim=c1)(student_f)


                        loss.append(self.MSELoss(teacher_f, student_f))
                product = list(map(lambda x, y: x * y, weight_1, loss))
                count_1 = 0.
                for i in range(len(product)):
                    count_1 += product[i].mean()
                count_1 /= len(product)
                count += count_1
                losses.append(count)

        return losses[0]*alpha + losses[1]*lamada +losses[2]*beta



# a = torch.randn(24,3,192,256)*20
# aa = torch.randn(24,3,192,256)*20
# b = torch.randn(24,50,96,128)*20
# bb = torch.randn(24,50,96,128)*20
# c = torch.randn(24,100,48,64)*20
# cc = torch.randn(24,100,48,64)*20
# d = torch.randn(24,100,48,64)*20
# e = torch.randn(24,100,48,64)*20
# teacher = [[a,b,c,d,e],[a,b,c,d,e],[a,b,c,d,e]]
# student = [[aa,bb,cc],[aa,bb,cc],[aa,bb,cc],]
#
# loss = FMDLoss()
#
# print(loss(teacher,student))




