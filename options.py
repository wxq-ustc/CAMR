# -*- coding: utf-8 -*-
import argparse
import os

import torch

### Parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/media/user/Disk 04/wxq/mydata2/', help="datasets")
    parser.add_argument('--datatype', default='cancer_lgg_val_data_1.pkl', help="datatype")
    parser.add_argument('--model_save', type=str, default='/media/user/Disk 04/wxq/mydata2/',help='models are saved here')
    parser.add_argument('--results', type=str, default='/media/user/Disk 04/wxq/mydata2/', help='results are saved here')
    parser.add_argument('--exp_name', type=str, default='1007', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_name', type=str, default='test', help='mode')
    parser.add_argument('--input_size', type=int, default=80, help="input_size for feature vector")
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting.')
    parser.add_argument('--hidden_size', default=512, type=int, help='the size of hidden units.')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.999, 0.5 | 0.25 | 0')
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=80, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=400, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--lr', default=0.0004, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lambda_reg', type=float, default=0.000002)

    parser.add_argument('--weight_decay', default=1e-6, type=float,help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--optimizer_type', type=str, default='adam')
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.model_save, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
