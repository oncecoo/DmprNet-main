import os
import sys
import time

import yaml
import logging

from easydict import EasyDict as edict

def set_config(args):
    #get configs file
    cfg_path = args.cfg_path

    #load configs file
    f = open(cfg_path,'r',encoding='utf-8')
    cont = f.read()
    x = yaml.full_load(cont)
    cfg = edict(x)

    return cfg
def set_T_config(args):
    #get configs file
    cfg_path = args.teacher_path

    #load configs file
    f = open(cfg_path,'r',encoding='utf-8')
    cont = f.read()
    x = yaml.full_load(cont)
    cfg = edict(x)

    return cfg
def set_logger(args,cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logger.prograte = False

    if cfg.save_log:
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(time.time() + 8 * 60 * 60))
        assert cfg.save_path != 'None'
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if args.model == 'train':
            log_name = 'log_train_%s_%s.log' % (cfg.dataset.name, time_str)

        elif args.model == 'eval':
            if 'resume_all_dir' in cfg.keys() and cfg.resume_all_dir != 'None':
                log_name = 'log_valAll_%s_%s.log'%(cfg.dataset.name,time_str)
            else:
                log_name = 'lpg_val_%s_%s.log'%(cfg.dataset.name,time_str)

        else:
            exit()
        os.makedirs(cfg.save_path + log_name,exist_ok=True)

        fh = logging.FileHandler(filename=os.path.join(cfg.save_path, log_name), mode='w', encoding=None,
                                     delay=False)

        logger.addHandler(fh)
    else:
        sh = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(sh)


    return  logger



def set_checkpoint(cfg,args,logger,include_dir=['configs','utils','models']):
    checkpoint = None
    if cfg.save_model:
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        checkpoint = cfg.save_path + '/checkpoint'

        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        else:
            error_msg = 'checkpoint dir already exist'
            logger.error(error_msg)
            raise ValueError(error_msg)

        code = cfg.save_path + '/code'
        if not os.path.exists(code):
            os.makedirs(code)
        else:
            error_msg = 'code dir already exist'
            logger.error(error_msg)
            raise ValueError(error_msg)


    return checkpoint