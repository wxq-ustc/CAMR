# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
#import adabound
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math
from fusion import LMF,CatFusion,Self_Attn,Atten,AttentionDot,Self_Attn1,mfb,GateFilter,MLPAttention,MCF,Core_Fusion,MLPAttention6,CatFusion2
def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        #optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1)
         pass
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    elif opt.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                      weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'sgd':
        torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

class MisaLMFGatedRec(nn.Module):
    def __init__(self,in_size, output_dim,hidden_size=20,hidden_size1=80,dropout=0.1):
        super(MisaLMFGatedRec, self).__init__()
      
        self.common = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                    nn.Linear(320, 128), nn.ReLU(),
                                    nn.Linear(128, 60), nn.ReLU())
        self.lmf = LMF(60, 16)
        self.unique1 = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                     nn.Linear(320, 60), nn.ReLU())
    
        self.unique2 = nn.Sequential(nn.Linear(in_size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 60), nn.ReLU())
       
        self.unique3 = nn.Sequential(nn.Linear(in_size, 260), nn.ReLU(),
                                     nn.Linear(260, 130), nn.ReLU(),
                                     nn.Linear(130, 60), nn.ReLU())
       
        encoder1 = nn.Sequential(nn.Linear(60 * 4, 900), nn.ReLU(), nn.Dropout(p=dropout))
       
        encoder2 = nn.Sequential(nn.Linear(900, 512), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(512,64), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(64,15), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(15, output_dim), nn.Sigmoid())
        self.output_range = Parameter(torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-4]), requires_grad=False)

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(60,60), nn.ReLU())
        self.linear_z1 = nn.Bilinear(60,120,60) 
        self.linear_o1 = nn.Sequential(nn.Linear(60,60), nn.ReLU(), nn.Dropout(p=dropout))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(60,60), nn.ReLU())
        self.linear_z2 = nn.Bilinear(60,120,60) 
        self.linear_o2 = nn.Sequential(nn.Linear(60,60), nn.ReLU(), nn.Dropout(p=dropout))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(60,60), nn.ReLU())
        self.linear_z3 = nn.Bilinear(60,120,60) 
        self.linear_o3 = nn.Sequential(nn.Linear(60,60), nn.ReLU(), nn.Dropout(p=dropout))

        ###recon
        self.rec1=nn.Sequential(nn.Linear(120,256),nn.ReLU(),
                                   nn.Linear(256,hidden_size1),nn.ReLU())

        self.rec2 = nn.Sequential(nn.Linear(120,256),nn.ReLU(),
                                   nn.Linear(256,hidden_size1),nn.ReLU())
        self.rec3 = nn.Sequential(nn.Linear(120,256),nn.ReLU(),
                                   nn.Linear(256,hidden_size1),nn.ReLU())
    def forward(self, x_gene, x_path, x_can):

        ##
        x_gene_common = self.common(x_gene)
        x_path_common = self.common(x_path)
        x_can_common = self.common(x_can)

        h1 = self.linear_h1(x_gene_common)
        vec31=torch.cat((x_path_common,x_can_common),
                               dim=1)
        z1 = self.linear_z1(x_gene_common, vec31)
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h2 = self.linear_h1(x_path_common)
        vec32 = torch.cat((x_gene_common, x_can_common),
                         dim=1)
        z2 = self.linear_z1(x_path_common, vec32)
        o2 = self.linear_o1(nn.Sigmoid()(z2) * h2)

        h3 = self.linear_h1(x_can_common)
        vec33 = torch.cat((x_gene_common, x_path_common),
                          dim=1)
        z3 = self.linear_z1(x_path_common, vec33)
        o3 = self.linear_o1(nn.Sigmoid()(z3) * h3)
        
        lmf=self.lmf(o1,o2,o3)
        ##
        x_gene_unique = self.unique1(x_gene)
        x_path_unique = self.unique2(x_path)
        x_can_unique = self.unique3(x_can)
      
        out_fusion = torch.cat((lmf, x_gene_unique, x_path_unique, x_can_unique),
                               dim=1)
        encoder = self.encoder(out_fusion)
        out = self.classifier(encoder)
        out = out * self.output_range + self.output_shift

        gene_rec=self.rec1(torch.cat((lmf,x_gene_unique),dim=1))
        path_rec = self.rec2(torch.cat((lmf,x_path_unique),dim=1))
        can_rec = self.rec3(torch.cat((lmf,x_can_unique),dim=1))
        return out, x_gene, gene_rec, x_path, path_rec, x_can, can_rec, x_gene_common, x_path_common, x_can_common,x_gene_unique,x_path_unique,x_can_unique
       