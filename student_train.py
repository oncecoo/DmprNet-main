import os.path
import os
import time

import torch
import random
import numpy as np
import argparse
from torch.utils import data
from utils.ScanNet_PlaneDataset import scannetv1_PlaneDataset
from utils.Loss_teacher import SetCriterion
from utils.utils import *
from utils.misc import *
from models.student_model import StudentNet
from models.teacher_model import Teacher_Net
from models.matcher import *
import torchvision.transforms as tf
from utils.Loss_distill import FMDLoss
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_path',default='configs/student_config.yaml',type=str)
parser.add_argument('--teacher_path',default='configs/teacher_config.yaml',type=str)

parser.add_argument('--model',default='train',type=str,help='train/help')

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'


def load_dataset(cfg,args):
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if args.model == 'train':
        subset = 'train'
    else:
        subset = 'val'
    is_shuffle = subset == 'train'


    if cfg.dataset.name == 'scannet':
        dataset = scannetv1_PlaneDataset(subset=subset,transform=transforms,root_dir=cfg.dataset.root_dir)
    else:
        print("undifined dataset!!")
        exit()


    loaders = data.DataLoader(
        dataset,batch_size=cfg.dataset.b_size,shuffle=is_shuffle,num_workers=cfg.dataset.num_workers,pin_memory=True
    )
    # print(len(dataset))
    return loaders

def train(cfg,logger):
    logger.info('*'*40)
    localtime = time.asctime(time.localtime(time.time()))
    logger.info(localtime)
    logger.info('strat training.......')
    logger.info('*'*40)

    model_name = (cfg.save_path).split('/')[-1]

    #set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    checkpoint_dir = set_checkpoint(cfg,args,logger)

    #--------------------------------------init network
    #----set if_distill
    if_distill = cfg.model.if_distill


    if if_distill == True:
        #Teacher model
        cfg_T = set_T_config(args)

        network_T = Teacher_Net(cfg_T)
        network_T.load_state_dict(torch.load("ckpts/teacher_network_epoch_49.pt", map_location=device))
        network_T.eval()
        network_T.train(mode=False)
        network_T = network_T.to(device)
        #Student model
        network_S = StudentNet()
        if os.path.exists("ckpts/student_network_epoch_49.pt"):
            network_S.load_state_dict(torch.load("ckpts/student_network_epoch_49.pt", map_location=device))
        network_S.eval()
        network_S.train()
        network_S = network_S.to(device)
        fmdloss = FMDLoss()
    else:
        #Student model
        network_S = StudentNet()
        network_S = network_S.to(device)
        network_S.eval()
        network_S.train(mode=False)

    #load pretrained weights of existed
    if not cfg.resume_dir == 'None':
        model_dict = torch.load(cfg.resume_dir)
        network_S.load_state_dict(model_dict)

    #---------------------------------------optimizer
    optimizer = get_optimizer(network_S.parameters(),cfg.solver)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.solver.lr_step, gamma=cfg.solver.gamma)

    # Loss_KD, lossCE
    lossCE = nn.CrossEntropyLoss()
    lossKLD = nn.KLDivLoss()

    #---------------------------------------dataloader
    loaders = load_dataset(cfg,args)
    #---------------------------------------set network state

    network_S.train(not cfg.model.fix_bn)

    k_inv_dot_xy1 = get_coordinate_map(device)
    matcher = HungarianMatcher(cost_class=1., cost_param=1.)


    weight_dict ={
        'loss_ce':1,
        'loss_pram_L1':1,
        'loss_pram_cos':5,
        'loss_embedding':5,
        'loss_Q':2,
        'loss_center_instance':1,
        'loss_center_pixel':1,
        'loss_depth_pixel':1,
        # 8
    }

    losses = ['labels','param','embedding','Q','center','depth']


    criterion_T = SetCriterion(num_classes=2,matcher = matcher ,weight_dict=weight_dict,eos_coef=1,
                               losses=losses,k_inv_dot_xy1=k_inv_dot_xy1)
    criterion_S = SetCriterion(num_classes=2,matcher = matcher ,weight_dict=weight_dict,eos_coef=1,
                               losses=losses,k_inv_dot_xy1=k_inv_dot_xy1)


    logger.info(f"used losses = {weight_dict}")
    start_epoch = 0
    T, lambda_stu = 5.0, 0.1
    beta = 10.
    for epoch in range(start_epoch,cfg.num_epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        metric_tracker = {'Classify_instance': ('loss_ce', AverageMeter()),
                          'Pull': ('loss_pull', AverageMeter()),
                          'Push': ('loss_push', AverageMeter()),
                          'PlaneParam_L1': ('loss_param_l1', AverageMeter()),
                          'PlaneParam_Cos': ('loss_param_cos', AverageMeter()),
                          'PlaneParam_Q': ('loss_Q', AverageMeter()),
                          'Center_Pixel': ('loss_center_pixel', AverageMeter()),
                          'Center_Plane': ('loss_center_instance', AverageMeter()),
                          'Depth_pixel': ('loss_depth_pixel', AverageMeter()),
                          'PlaneParam_Angle': ('mean_angle', AverageMeter())}    # 10
        tic = time.time()
        for iter,sample in enumerate(loaders):
            image = sample['image'].to(device)  # b, 3, h, w
            instance = sample['instance'].to(device)
            # semantic = sample['semantic'].to(device)
            gt_depth = sample['depth'].to(device)  # b, 1, h, w
            gt_seg = sample['gt_seg'].to(device)
            gt_plane_parameters = sample['plane_parameters'].to(device)
            valid_region = sample['valid_region'].to(device)
            gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)#b，20，3
            # print(gt_plane_instance_parameter.shape)
            gt_plane_instance_centers = sample['gt_plane_instance_centers'].to(device)
            gt_plane_pixel_centers = sample['gt_plane_pixel_centers'].to(device)
            num_planes = sample['num_planes']
            torch.cuda.empty_cache()

            if if_distill == True:
                t1,t2,t3,outputs_T = network_T(image)
                s1, s2, s3, outputs_S = network_S(image)
            else:
                s1, s2, s3, outputs_S = network_S(image)

            targets = []
            batch_size = image.size(0)
            for i in range(batch_size):
                gt_plane_num = int(num_planes[i])
                tgt = torch.ones([gt_plane_num, 6], dtype=torch.float32, device=device)
                tgt[:, 0] = 1
                tgt[:, 1:4] = gt_plane_instance_parameter[i, :gt_plane_num, :]
                tgt[:, 4:6] = gt_plane_instance_centers[i, :gt_plane_num, :]
                # tgt[:, 6:9] = gt_plane_instance_normal[i,:gt_plane_num,:]
                #加入新的维度：法线
                tgt = tgt.contiguous().cuda()
                targets.append(tgt)
                torch.cuda.empty_cache()

            if if_distill == False:
                outputs_S['gt_instance_map'] = instance
                outputs_S['gt_depth'] = gt_depth
                outputs_S['valid_region'] = valid_region

                # calculate losses
                loss_dict, _ = criterion_S(outputs_S, targets)

                loss_final = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # --------------------------------------  Backward
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                # --------------------------------------  update losses and metrics
                losses.update(loss_final)

                for name_log in metric_tracker.keys():
                    name_loss = metric_tracker[name_log][0]
                    if name_loss in loss_dict.keys():
                        loss_cur = float(loss_dict[name_loss])
                        metric_tracker[name_log][1].update(loss_cur)
                # ------------------------------------------------update time
                batch_time.update(time.time() - tic)
                tic = time.time()

                # ------------------------------------------------log append info _ bs
                if iter % cfg.print_interval == 0:
                    # print(data_path)
                    # log_str = f"[{epoch.val:2d}][{iter.val:5d}/] " \
                    #           f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "
                    #           # f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    #
                    # for name_log, (_, tracker) in metric_tracker.items():
                    #     log_str += f"{name_log}: {tracker.val:.4f} ({tracker.avg:.4f}) "
                    # logger.info(log_str)

                    print(f"[{model_name}-> {epoch:2d}][{iter:5d}/] "
                          f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) ")
                          # f"Loss: {losses.val:.4f} ({losses.avg:.4f}) ")
                    logger.info('-------------------------------------')


            elif if_distill == True:
                outputs_T['gt_instance_map'] = outputs_S['gt_instance_map'] = instance
                outputs_T['gt_depth'] = outputs_S['gt_depth'] =  gt_depth
                # outputs_T['gt_plane_pixel_centers'] = outputs_S['gt_plane_pixel_centers'] = gt_plane_pixel_centers
                outputs_T['valid_region'] = outputs_S['valid_region'] = valid_region

                outputs_T['gt_instance_map'] = outputs_S['gt_instance_map'] = instance
                outputs_T['gt_depth'] = outputs_S['gt_depth'] = gt_depth
                # outputs_T['gt_plane_pixel_centers'] = outputs_S['gt_plane_pixel_centers'] = gt_plane_pixel_centers
                outputs_T['valid_region'] = outputs_S['valid_region'] = valid_region

                loss_dict, _ = criterion_T(outputs_T, targets)
                loss_dict_S, _ = criterion_S(outputs_S, targets)
                loss_kd = fmdloss([t1,t2,t3],[s1,s2,s3])

                # loss_teacher
                loss_final = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # loss_student

                loss_final_S = sum(loss_dict_S[k] * weight_dict[k] for k in loss_dict_S.keys() if k in weight_dict)

                dic = ['pred_logits', 'pred_param', 'pred_plane_embedding', 'pixel_embedding', 'pred_center',
                       'pixel_depth']
                loss_KLD = 0.
                for i in range(len(dic)):

                    loss_KLD += lambda_stu + lambda_stu * lossKLD(outputs_S[dic[i]], outputs_T[dic[i]])

                loss = lambda_stu * loss_final_S + (1 - lambda_stu) * loss_final + beta*loss_kd + loss_KLD


                losses.update(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # -------------------------------update time
                batch_time.update(time.time() - tic)
                tic = time.time()

                # ------------------------------------------------log append info _ bs
                if iter % cfg.print_interval == 0:
                    # print(data_path)
                    log_str = f"[{epoch:2d}][{iter:5d}/{len(loaders) :5d}] " \
                              f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) "\
                              # f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "

                    for name_log, (_, tracker) in metric_tracker.items():
                        log_str += f"{name_log}: {tracker.val:.4f} ({tracker.avg:.4f}) "
                    logger.info(log_str)

                    print(f"[{model_name}-> {epoch:2d}][{iter:5d}/] "
                          f"Time: {batch_time.val:.2f} ({batch_time.avg:.2f}) ")
                          # f"Loss: {losses.val:.4f} ({losses.avg:.4f}) ")
                    logger.info('-------------------------------------')


        torch.cuda.empty_cache()

        lr_scheduler.step()

        # log for one epoch
        logger.info('*' * 40)

        log_str = f"[{epoch:2d}] " \
                  f"Loss: {losses.avg:.4f} "
        for name_log, (_, tracker) in metric_tracker.items():
            log_str += f"{name_log}: {tracker.avg:.4f} "
        logger.info(log_str)
        logger.info('*' * 40)

        # save checkpoint
        if cfg.save_model :
            if (epoch + 1) % cfg.save_step == 0 or epoch >= 58:
                torch.save(network_S.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


if __name__ == '__main__':
    cfg = set_config(args)
    logger = set_logger(args,cfg)
    if args.model == 'train':
        train(cfg,logger)
    else:
        exit()