import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from models.HRNet import build_backbone
from models.transformer import *
from models.position_encoding import build_position_encoding

logger = logging.getLogger(__name__)
use_biase = False
use_align_corners = False

def conv_bn_relu(in_chan,out_chan,k=1,pad=0):
    '''
        This is a conv2d block
    '''
    return nn.Sequential(
        nn.Conv2d(in_chan,out_chan,(k,k)),
        nn.BatchNorm2d(out_chan),
        nn.ReLU(inplace=True),
    )

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


class top_down(nn.Module):
    def __init__(self, in_channels=[], channel=64, m_dim=256, double_upsample=False):
        super(top_down, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.double_upsample = double_upsample

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=use_align_corners)
        if double_upsample:
            self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=use_align_corners)
        self.up_conv4 = conv_bn_relu(channel, channel, 1)
        self.up_conv3 = conv_bn_relu(channel, channel, 1)
        self.up_conv2 = conv_bn_relu(channel, channel, 1)
        self.up_conv1 = conv_bn_relu(channel, channel, 1)

        # lateral
        self.hidden_dim = in_channels[3]
        self.c4_conv = conv_bn_relu(in_channels[3], channel, 1)
        self.c3_conv = conv_bn_relu(in_channels[2], channel, 1)
        self.c2_conv = conv_bn_relu(in_channels[1], channel, 1)
        self.c1_conv = conv_bn_relu(in_channels[0], channel, 1)

        self.m_conv_dict = nn.ModuleDict({})
        self.m_conv_dict['m4'] = conv_bn_relu(m_dim, channel)

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.conv_tran = nn.Sequential(
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(1,1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3,3), stride=(2,2),padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(1,1)),
        )

    def init_weights(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, memory):
        c1, c2, c3, c4 = x

        f1 = self.conv_tran(c4)
        p5 = self.c4_conv(f1) + self.m_conv_dict['m4'](memory)

        p4 = self.up_conv4(self.upsample(p5)) + self.c4_conv(c4)

        p3 = self.up_conv3(self.upsample(p4)) + self.c3_conv(c3)

        p2 = self.up_conv2(self.upsample(p3)) + self.c2_conv(c2)

        p1 = self.up_conv1(self.upsample(p2)) + self.c1_conv(c1)

        if self.double_upsample:
            p0 = self.upsample2(p1)
        else:
            p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4



class Teacher_Net(nn.Module):
    def __init__(self,cfg,position_embedding_mode = 'sine'):
        super(Teacher_Net, self).__init__()
        num_queries = cfg.model.num_queries
        plane_embedding_dim = cfg.model.plane_embedding_dim
        loss_layer_num = cfg.model.loss_layer_num

        #---------------------------------feature extractor
        self.backbone = build_backbone(cfg.model)
        self.backbone_channels = self.backbone.out_channels

        #---------------------------------pre_define
        self.loss_layer_num = loss_layer_num
        assert self.loss_layer_num <= 2
        self.return_inter = False

        self.num_sample_pts = cfg.model.num_sample_pts
        self.if_predict_depth = cfg.model.if_predict_depth
        self.if_shareHeads = True
        assert cfg.model.stride == 1

        #network super-param
        self.channels = 64
        self.hidden_dim = 256
        self.num_queries = num_queries
        self.context_channels = self.backbone_channels[-1]   #num =

        self.plane_embedding_dim = plane_embedding_dim

        #------------------------------------------------------------------------------------transformer branch
        self.input_proj = nn.Conv2d(self.context_channels,self.hidden_dim,(1,1))

        self.position_embedding = build_position_encoding(position_embedding_mode=position_embedding_mode,hidden_dim=self.hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries,self.hidden_dim) # 服从正态分布的（0，1）的随机一组数
        self.transformer = build_transformer(hidden_dim=self.hidden_dim,dropout=0.1,nheads=8,dim_feedforward=1024,
                                             enc_layers=6,dec_layers=cfg.model.dec_layers,pre_norm=True,return_inter=self.return_inter)

        #---------------------instance-level plane embedding
        self.plane_embedding =  MLP(self.hidden_dim,self.hidden_dim,self.plane_embedding_dim,3)
        #---------------------planar / non-planar classification
        self.plane_prob=nn.Linear(self.hidden_dim,3)
        #---------------------instance -level plane param
        self.plane_param = MLP(self.hidden_dim,self.hidden_dim,3,3)
        #---------------------instance-level plane center
        self.plane_center = MLP(self.hidden_dim,self.hidden_dim,2,3)

        #--------------------------------------------------------------------------------------convolution branch
        #top_dow
        self.top_down = top_down(self.backbone_channels,self.channels,m_dim=self.hidden_dim,double_upsample=False)
        #pixel embedding
        self.pixel_embedding = nn.Conv2d(self.channels,self.plane_embedding_dim,(1,1),padding=0)
        #pixel-level plane center

        self.pixel_plane_center = nn.Conv2d(self.channels,2,(1,1),padding=0)      #channel = 2
        #pixel-level plane depth

        self.top_down_depth = top_down(self.backbone_channels,self.channels,m_dim=self.hidden_dim,double_upsample=False)
        self.depth = nn.Conv2d(self.channels,1,(1,1),padding=0)

        self.conv_tran1 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,2),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=(1,1)),
        )
        self.conv_tran2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2,2),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=(1,1)),
        )
        self.conv_tran3 = nn.Sequential(
            nn.Conv2d(128,self.hidden_dim,kernel_size=(1,1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3,3), stride=(2,2),padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(1,1)),
        )
        self.conv_tran4 = nn.Sequential(
            nn.Conv2d(256,self.hidden_dim,kernel_size=(1,1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3,3), stride=(2,2),padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=(1,1)),
        )

    def forward(self,x):
        b,_,h,w = x.shape
        c1,c2,c3,c4 = self.backbone(x)  # c: 32, 64, 128, 256

        f1= self.conv_tran4(self.conv_tran3(self.conv_tran2(self.conv_tran1(c1)+c2)+c3)+c4)
        print('******',f1.shape)
        #context src feature
        src = f1

        #context feature proj
        src = self.input_proj(src)

        #position embedding
        pos1 = self.position_embedding(src)


        #---------------------------------------------------------------------------transformer

        hs_all,_,memory = self.transformer(src,self.query_embed.weight,pos1,tgt=None,src_lines = None,
                                            mask_lines = None,pos_embed_lines= None) # memory: b, c, h, w


        #--------------------------------------------------------------------------plane instance decoder
        hs = hs_all[-self.loss_layer_num:, :, :, :].contiguous()  # dec_layers, b, num_queries, hidden_dim
        #plane embedding
        plane_embedding = self.plane_embedding(hs)
        # print('plane_embedding_shape',plane_embedding.shape)
        #plane classifier
        plane_prob = self.plane_prob(hs)
        # print('plane_prob',plane_prob.shape)
        #plane param
        plane_param = self.plane_param(hs)
        # print('plane_param',plane_param.shape)
        #plane_center

        plane_center = self.plane_center(hs)
        plane_center = torch.sigmoid(plane_center)

        #---------------------------------------------------------------------------pixel-level decoder
        p0,p1,p2,p3,p4 = self.top_down((c1,c2,c3,c4),memory)
        pixel_embedding = self.pixel_embedding(p0)
        # print('pixel_embedding_shape',pixel_embedding.shape)

        pixel_center = self.pixel_plane_center(p0)
        pixel_center = torch.sigmoid(pixel_center)

        p_depth,_,_,_,_ = self.top_down_depth((c1,c2,c3,c4),memory)
        pixel_depth = self.depth(p_depth)

        #---------------------------------------------------------------------------------------------output
        output = {
            'pred_logits':plane_prob[-1],
            'pred_param':plane_param[-1],
            'pred_plane_embedding':plane_embedding[-1],
            'pixel_embedding':pixel_embedding
        }

        output['pixel_center'] = pixel_center
        output['pred_center'] = plane_center[-1]

        output['pixel_depth'] = pixel_depth


        #
        # print('pred_logits',output['pred_logits'].shape)
        # print('pred_param',output['pred_param'].shape)
        # print('pred_plane_embedding',output['pred_plane_embedding'].shape)
        # print('pixel_embedding',output['pixel_embedding'].shape)
        # print('pixel_center',output['pixel_center'].shape)
        # print('pred_center',output['pred_center'].shape)
        # print('pixel_depth',output['pixel_depth'].shape)
        # print('aux_outputs',output['aux_outputs'].shape)

        return [c1,c2,c3,c4],[f1,memory],[p4,p3,p2,p1,p0],output

# import argparse
# from utils.utils import *
#
# a = torch.randn(24,3,192,256)
# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg_path',default='../configs/teacher_config.yaml',type=str)
# parser.add_argument('--model',default='train',type=str,help='train/help')
# parser.add_argument('--backbone', default='hrnet', type=str,
#                     help='only support hrnet now')
# args = parser.parse_args()
# cfg = set_config(args)
#
# model = Teacher_Net(cfg)
# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
# # aa,bb,cc,dd = model(a)
# print(len(aa),len(bb),len(cc))
#
# for i in range(len(aa)):
#     print(aa[i].shape)
# print('*'*50)
#
# for i in range(len(bb)):
#     print(bb[i].shape)
# print('*'*50)
# for i in range(len(cc)):
#     print(cc[i].shape)
