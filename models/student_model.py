import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.mobilenetV3 import MobileNetV3


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def conv_bn(in_channels,out_channles,k,stride,groups=1):
    padding = (k-1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channles,(k,k),stride=stride,padding=padding,groups=groups),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channles),

    )

def conv_dil(in_channels,out_channles,k,dilation,bias=False):
    padding = (k-1)//2
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channles,(k,k),stride=1,padding=padding,bias=bias),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channles),)






class Topdown(nn.Module):
    def __init__(self, in_channels=[], channel=8):
        super(Topdown, self).__init__()
        self.relu = nn.ReLU(inplace=True)


        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.up_conv = conv_bn(channel, channel, 1,1)


        # lateral
        self.c6_conv = conv_bn(in_channels[0], channel, 1,1)    #100 |
        self.c5_conv = conv_bn(in_channels[1], channel, 1,1)    #100 |
        self.c4_conv = conv_bn(in_channels[2], channel, 1,1)    #48  | 80
        self.c3_conv = conv_bn(in_channels[3], channel, 1,1)    #24  | 40
        self.c2_conv = conv_bn(in_channels[4], channel, 1,1)    #16  | 24
        self.c1_conv = conv_bn(in_channels[5], channel, 1,1)    #16  | 16


        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

    def forward(self, x):
        c1, c2, c3, c4, c5, c6 = x
        p6 = self.c6_conv(c6)

        # print(p6.shape,self.c5_conv(c5).shape)
        p5 = self.up_conv(p6 + self.c5_conv(c5))
        p4 = self.up_conv(self.upsample(p5)) + self.c4_conv(c4)
        p3 = self.up_conv(self.upsample(p4)) + self.c3_conv(c3)
        p2 = self.up_conv(self.upsample(p3)) + self.c2_conv(c2)
        p1= self.up_conv(self.upsample(p2)) + self.c1_conv(c1)

        p0 = self.upsample(p1)
        p0 = self.relu(self.p0_conv(p0))

        return p0

class SED(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SED, self).__init__()

        self.upsampling = nn.Upsample(scale_factor=2,mode='bilinear')
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv_init = nn.Conv2d(in_channels,in_channels,kernel_size=1,bias=False)
        self.conv_dil5 = conv_dil(in_channels,out_channels,k=7,dilation=5,bias=True)
        self.conv_dil7 = conv_dil(in_channels,out_channels,k=7,dilation=7,bias=True)
        self.conv_dil9 = conv_dil(in_channels,out_channels,k=7,dilation=9,bias=True)
        self.hidden_dim = 8

        #self.feature_filtering = nn.Sequential(
        #    nn.Linear(out_channels,self.hidden_dim),
        #     nn.ReLU(True),
        #    nn.Dropout(0.1),
        #     nn.Linear(self.hidden_dim,out_channels)
        # )
        self.pool = nn.Sequential(
            self.maxpool,
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1),
        )
    def forward(self,x):
        x = self.relu(self.conv_init(x))
        x = self.upsampling(x)
        out = self.conv_dil5(x)+self.conv_dil7(x)+self.conv_dil9(x)
        out = self.relu(out)
        # _,c,h,w = out.shape
        # # print(out.shape)
        # out = out.reshape(-1,h*w,c)
        # # print('*'*20,out.shape)
        #out = self.feature_filtering(out)
        # print('*'*20,out.shape)
        #
        # out = out.reshape(-1,c,h,w)

        return self.pool(out)




class FeatureFusion(nn.Module):
    'the goal channels is c2,and the c1 is the num_classes:1000'
    def __init__(self,in_channels,out_channels):
        super(FeatureFusion, self).__init__()
        self.num_classes = in_channels
        self.out_channels = out_channels

        self.conv_rem = conv_bn(in_channels=self.out_channels,out_channles=self.out_channels,k=1,stride=1)
        self.conv_tran = conv_bn(in_channels=self.num_classes,out_channles=self.out_channels,k=1,stride=1)
        self.conv_block1 = nn.Sequential(
            conv_bn(in_channels=self.out_channels,out_channles=self.out_channels,k=1,stride=1),
            conv_bn(in_channels=self.out_channels, out_channles=self.out_channels, k=7, stride=1),
            conv_bn(in_channels=self.out_channels, out_channles=self.out_channels, k=1, stride=1)
        )
        self.conv_red = conv_bn(in_channels=self.out_channels,out_channles=self.out_channels,k=1,stride=1)

        self.SED = SED(self.num_classes,self.num_classes)

    def forward(self,x,y):
        # print('x.shape',x.shape)
        x = self.SED(x)
        # print(x.shape)
        _,c1,h1,w1 = x.size()
        _,c2,h2,w2 = y.size()
        if h1 != h2 and w1 != w2:
            raise('Tensors are not uniform in size!!!')
        output = self.conv_red(self.conv_block1(self.conv_tran(x))+self.conv_rem(y))
        # print(output.shape)
        return output






class StudentNet(nn.Module):
    def __init__(self,model_mode="LARGE",num_classes=100):
        super(StudentNet, self).__init__()
        self.model_mode = model_mode
        self.num_classes = num_classes
        self.out_channel = 16
        self.encoder = MobileNetV3()
        # self.in_channels = [self.num_classes,256,128,64,32,16]
        self.in_channels = [self.num_classes,160,80,40,24,16]

        self.classes_large = Topdown(in_channels=self.in_channels)

        self.hidden_dim = 256
        self.plane_embedding_dim = 8
        self.low_channel = 8
        self.require_num = 20

        self.featurefusion_256 = FeatureFusion(in_channels=self.in_channels[0],out_channels=self.in_channels[1])
        self.featurefusion_128 = FeatureFusion(self.in_channels[1],self.in_channels[2])
        self.featurefusion_64 = FeatureFusion(self.in_channels[2],self.in_channels[3])
        self.featurefusion_32 = FeatureFusion(self.in_channels[3],self.in_channels[4])
        self.featurefusion_16 = FeatureFusion(self.in_channels[4],self.in_channels[5])
        self.featurefusion_8 = FeatureFusion(self.in_channels[5],8)


        self.upsampling = nn.Upsample(scale_factor=2,mode='bilinear')


        self.pred_logit_trans = MLP(1,self.hidden_dim,3,2)
        self.pred_param_trans = MLP(1,self.hidden_dim,3,3)
        self.pred_embedding_trans =  MLP(1,self.hidden_dim,self.plane_embedding_dim,3)
        self.plane_center = MLP(2,self.hidden_dim,2,3)
        self.pred_surface_normal = conv_bn(self.low_channel,3,1,1)
        self.pixel_embedding = nn.Conv2d(self.low_channel,self.plane_embedding_dim,(1,1),padding=0)
        self.non_plane_embedding = conv_bn(self.low_channel,1,1,1)
        #self.pixel_center = conv_bn(self.low_channel,2,1,1)

        self.depth = nn.Conv2d(self.low_channel, 1, (1, 1), padding=0)

        self.conv1 = nn.Conv2d(num_classes,48,(1, 1), padding=0)
        self.conv2 = nn.Conv2d(num_classes,24,(1, 1), padding=0)
        self.conv3 = nn.Conv2d(num_classes,16,(1, 1), padding=0)
        self.conv_rem = nn.Conv2d(self.low_channel,self.low_channel,(1,1),padding=0)
        self.conv1_rem = nn.Conv2d(in_channels=self.low_channel,out_channels=self.require_num,kernel_size=(1,1),padding=0)

        self.conv_down = nn.Conv2d(self.out_channel,self.low_channel,(1,1),padding=0)


    def forward(self,x):


        x0,x1,x2,x3,x4,x5 = self.encoder(x)
        p0 = self.classes_large(self.encoder(x))#15, 8, 192, 256
        # print(p0.shape)
        output1 = x5
        # print(output1.shape,x4.shape)
        result1 = self.featurefusion_256(output1,x4)
        # print('result1',result1.shape)
        output2 = self.upsampling(result1)
        # print(output2.shape,x3.shape)
        result2 = self.featurefusion_128(output2,x3)
        # print(result2.shape)
        output3 = self.upsampling(result2)
        result3 = self.featurefusion_64(output3,x2)
        # print(result3.shape)
        output4 = self.upsampling(result3)
        result4 = self.featurefusion_32(output4,x1)# b, 32, 32, 40
        # print('result4',result4.shape)
        output5 = self.upsampling(result4) # b, 32, 64, 80
        result5 = self.featurefusion_16(output5,x0)# b, 16, 64, 80
        # print(result5.shape)
        output6 = self.upsampling(result5)# b, 16, 128, 160

        x = self.conv_down(self.upsampling(x0))   # b, 8, 128, 160

        # print('1',output6.shape,x.shape)
        result5 = self.featurefusion_8(output6,x)

        result = self.conv_rem(self.conv_rem(result5)* self.conv_rem(p0))
        # print('result.shape',result.shape)

        pred_logits = self.conv1_rem(F.adaptive_avg_pool2d(result,(1,1))).squeeze(-1)
        pred_param =self.conv1_rem(F.adaptive_avg_pool2d(result,(1,1))).squeeze(-1)
        pred_plane_embedding = self.conv1_rem(F.adaptive_avg_pool2d(result,(1,1))).squeeze(-1)
        plane_center = self.conv1_rem(F.adaptive_avg_pool2d(result,(3,2))).squeeze(-1)
        pixel_depth = self.depth(result)
        pixel_embedding = self.conv_rem(result)
        # print('plane_center',plane_center.shape)

        output ={
            'pred_logits': self.pred_logit_trans(pred_logits),                      #25, 20, 3
            'pred_param': self.pred_param_trans(pred_param) ,                       #25, 20, 3
            'pred_plane_embedding': self.pred_embedding_trans(pred_plane_embedding),#25, 20, 8
            'pixel_embedding': pixel_embedding,                                     #25, 8, 192, 256
            'pred_center':torch.mean(self.plane_center(plane_center),dim=2),        #25, 20, 2
            'pixel_depth' : pixel_depth,                                            #25, 1, 192, 256
            'pred_surface_normal': self.pred_surface_normal(result),

        }


        return [x0,x1,x2,x3],[x4,x5,result1],[result2,result3,result4,result5,result],output





# a = torch.randn(25, 3, 192, 256)
# model = StudentNet()
# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# x1,x2,x3,x4 = model(a)
# out1,out2,out3,out4,out5= model(a)
# #
# a,b,c,p = model(a)
# print(len(p))
#
# for i in range(len(a)):
#     print(a[i].shape)
# print('*'*50)
# for i in range(len(b)):
#     print(b[i].shape)
# print('*'*50)
# for i in range(len(c)):
#     print(c[i].shape)
# #
# print("result is",out1.shape,out2.shape,out3.shape,out4.shape,out5.shape)
# print('p0',p.shape)
# print('pred_logits',p['pred_logits'].shape,'\n',
#      'pred_param',p['pred_param'].shape,'\n',
#      'pred_plane_embedding',p['pred_plane_embedding'].shape,'\n',
#     'pixel_embedding',p['pixel_embedding'].shape,'\n',
#      'pixel_depth',p['pixel_depth'].shape,'\n',
#      'pred_surface_normal',p['pred_surface_normal'].shape,'\n',
#       # 'pixel_center',p['pixel_center'].shape,'\n',
#      'pred_center', p['pred_center'].shape)
# x1,x2,x3,x4,x5 = model(a)
# print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

