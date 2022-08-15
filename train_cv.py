# -*- coding: utf-8 -*-
import os

import logging
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch
import pandas as pd
# Env
from data_loaders import *
from options import parse_args
from train_test import train, test



opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
print(torch.cuda.device_count())  # 打印gpu数量
if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
        os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

data_cv_path = '%s%s' % (opt.dataroot,opt.datatype)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results=[]

average_results = []
best_results = []
os_time,os_status,risk_pred = [],[],[]
best_os_time,best_os_status,best_risk_pred = [],[],[]

code_pred =[]
label_pred = []
### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")

    ### ### ### ### ### ### ### ### ###创建文件夹存储结果### ### ### ### ### ### ### ### ### ###
    if not os.path.exists(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k))): os.makedirs(
        os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k)))

    ### 3.1 Trains Model
    model, optimizer, metric_logger = train(opt, data, device, k)
    epochs_list = range(opt.epoch_count, opt.niter+opt.niter_decay+1)
    ### 3.2 Evalutes Train + Test Error, and Saves Model
    loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train= test(opt,model,data, 'train',device)
    loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test=test(opt, model, data, 'test', device)

   
    print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
  
    average_results.append(cindex_test)
    ### 3.3 Saves Model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        'split':k,
        'opt': opt,
        'epoch': opt.niter+opt.niter_decay,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        os.path.join(opt.model_save, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
    )
    print()

    pickle.dump(pred_train, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
    pickle.dump(pred_test, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))
    df = pd.DataFrame({'os_time': pred_test[1], 'os_status': pred_test[2], 'risk_pred': pred_test[0]})
    df.to_csv(opt.results + "%d-fold_pred.csv" % (k), index=0, header=1)

    PI = pred_test[0] > np.median(pred_test[0])
    np.savetxt(opt.results + "%d-fold_label_test.csv" % (k), PI + 0, delimiter=",")
    risk_pred.extend(pred_test[0])
    os_time.extend(pred_test[1])
    os_status.extend(pred_test[2])
    label_pred.extend(PI + 0)

df = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':risk_pred})
df.to_csv(opt.results + "out_pred_5fold.csv", index=0, header=1)

df2 = pd.DataFrame({'os_time':os_time,'os_status':os_status,'risk_pred':label_pred})
df2.to_csv(opt.results + "risk_pred_5fold.csv", index=0, header=1)

np.savetxt(opt.results + "label_test.csv", label_pred, fmt="%d",delimiter=",")
np.savetxt(opt.results + "split_average_results.csv", average_results, delimiter=",")
print('Split Average Results:', average_results)
print('Split Best Results:', best_results)
print("Average_results:", np.array(average_results).mean()," std: ", np.std(average_results,ddof = 0))
pickle.dump(average_results, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))