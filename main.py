# %%导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import os
import argparse
from tester import TranAD_tester, AE_basic_tester, GRELEN_tester, COUTA_tester, HSR_tester,SSHSR_tester,GDN_tester,MTAD_GAT_tester
from util import LoadConfig, metric, pot, roc_auc_score, Set_Seed, Save_Model, ploting, slice_windows_data, get_metrics, \
    adjust_scores,produce_train_target_data,GDN_proceese_data
from dataset import TranAD_dataset, AE_basic_dataset, GRELEN_dataset, COUTA_dataset, HSR_dataset,SSHSR_dataset,GDN_dataset,MTAD_GAT_dataset
from torch.utils.data import DataLoader
from models import TranAD_model as TranAD_model
from models import HSR_model_2 as HSR_model
from models import AE_basic, GRELEN_model, COUTA_model,SSHSR_model,GDN_model,MTAD_GAT_model
from trainer import TranAD_trainer, AE_basic_trainer, GRELEN_trainer, COUTA_trainer, HSR_trainer2,HSR_trainer,SSHSR_trainer, GDN_trainer,MTAD_GAT_trainer
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler

# %%设置参数

with open('config/HSR/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(config[config['dataset']])
    del config[config['dataset']]
# 固定随机数
Set_Seed.__setseed__(config['random_seed'])

device = config['device']

# %% 设定dataset
if config['model'] == 'TranAD':
    train_dataset_w = TranAD_dataset.TranADdataset_W(config, flag='train', nosie=False)
    if config['dataset'] == 'penism':#在这个模型中，只有penism有验证集，其他的就没有验证集了
        val_dataset_w = TranAD_dataset.TranADdataset_W(config, flag='val')
    test_dataset_W = TranAD_dataset.TranADdataset_W(config, flag='test')
    if config['dataset'] == 'penism':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        val_data_orig = np.loadtxt(config['val_data_path'], delimiter=',')
        train_val_data_orig, val_data_orig, test_data_orig = train_data_orig[:8000, :], val_data_orig[:,:-1], test_data_orig[:,:-1]  # 把这个提出来是因为后面绘图要用
        labels = np.loadtxt(config['test_data_path'], delimiter=',')[:, -1]

    if config['dataset'] == 'MSL':
        test_data_orig = np.load(config['test_data_path'])
        train_data_orig = np.load(config['train_data_path'])
        labels = np.load(config['label_path'])
    if config['dataset'] == 'WADI':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        train_val_data_orig, test_data_orig = train_data_orig, test_data_orig  # 把这个提出来是因为后面绘图要用
        labels = np.loadtxt(config['label_path'], delimiter=',')
    if config['dataset'] == 'SMD':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        train_val_data_orig = train_data_orig  # 把这个提出来是因为后面绘图要用
        labels = np.loadtxt(config['label_path'], delimiter=',')
if config['model'] == 'AE_basic':
    train_dataset = AE_basic_dataset.dataset(config, flag='train')
    val_dataset = AE_basic_dataset.dataset(config, flag='val')
    test_dataset = AE_basic_dataset.dataset(config, flag='test')
    train_val_dataset = AE_basic_dataset.dataset(config, flag='train_val')
    if config['dataset'] == 'penism':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        val_data_orig = np.loadtxt(config['val_data_path'], delimiter=',')
        train_val_data_orig, val_data_orig, test_data_orig = train_data_orig[:8000, :], val_data_orig[:,
                                                                                        :-1], test_data_orig[:, :-1]
        labels = np.loadtxt(config['test_data_path'], delimiter=',')[:, -1]
if config['model'] == 'GRELEN':
    train_dataset = GRELEN_dataset.dataset(config, flag='train')
    test_dataset = GRELEN_dataset.dataset(config, flag='test')
    train_val_dataset = GRELEN_dataset.dataset(config, flag='train_val')
    if config['dataset'] == 'SWAT':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = np.loadtxt(config['label_path'], delimiter=',')
        res = len(labels) % config['window_size']
        labels = labels[:-res]
    if config['dataset'] == 'penism':
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')[:,:-1]
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = np.loadtxt(config['test_data_path'], delimiter=',')[:, -1]
        res = len(labels) % config['window_size']
        labels = labels[:-res]#去掉最后的label的部分
    if config['dataset'] == 'WADI':
        test_data_orig = np.loadtxt(config['test_data_path'],delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'],delimiter=',')
        labels = np.loadtxt(config['label_path'],delimiter=',')
        res = len(labels) % config['window_size']
        if res != 0:
            labels = labels[:-res]
    if config['dataset'] == 'SMD':
        test_data_orig = np.loadtxt(config['test_data_path'],delimiter=',')
        train_data_orig = np.loadtxt(config['train_data_path'],delimiter=',')
        labels = np.loadtxt(config['label_path'],delimiter=',')
        res = len(labels) % config['window_size']
        labels = labels[:-res]
    if config['dataset'] == 'MSL':
        test_data_orig = np.load(config['test_data_path'])
        train_data_orig = np.load(config['train_data_path'])
        labels = np.load(config['label_path'])
        labels = (np.sum(labels, axis=1) >= 1) + 0
        res = len(labels) % config['window_size']
        if res!=0:
            labels = labels[:-res]
if config['model'] == 'COUTA':
    if config['dataset'] == 'SWAT':
        # 把数据进行切片和val的划分
        train_data = np.loadtxt(config['train_data_path'], delimiter=',')
        test_data = np.loadtxt(config['test_data_path'], delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 100
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        # 划分val
        val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        val_data = data[val_seletcion, :, :]
        train_data = np.delete(data, val_seletcion, axis=0)
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        val_dataset = COUTA_dataset.dataset(config, val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        # train_val_dataset = COUTA_dataset.dataset(config,train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = np.loadtxt(config['label_path'], delimiter=',')
    if config['dataset'] == 'penism':
        train_data = np.loadtxt(config['train_data_path'], delimiter=',')
        test_data = np.loadtxt(config['test_data_path'], delimiter=',')[:, :-1]
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_val_data = data
        # 划分val
        val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        val_data = data[val_seletcion, :, :]
        train_data = np.delete(data, val_seletcion, axis=0)
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        val_dataset = COUTA_dataset.dataset(config, val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        train_val_dataset = COUTA_dataset.dataset(config, train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = test_data_orig[:, -1]
    if config['dataset'] == 'WADI':
        train_data = np.loadtxt(config['train_data_path'], delimiter=',')
        test_data = np.loadtxt(config['test_data_path'], delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        # 划分val
        val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        val_data = data[val_seletcion, :, :]
        train_data = np.delete(data, val_seletcion, axis=0)
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        val_dataset = COUTA_dataset.dataset(config, val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        # train_val_dataset = COUTA_dataset.dataset(config,train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = np.loadtxt(config['label_path'], delimiter=',')
    if config['dataset'] == 'SMD':
        train_data = np.loadtxt(config['train_data_path'], delimiter=',')
        test_data = np.loadtxt(config['test_data_path'], delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        # 划分val
        val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        val_data = data[val_seletcion, :, :]
        train_data = np.delete(data, val_seletcion, axis=0)
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        val_dataset = COUTA_dataset.dataset(config, val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        # train_val_dataset = COUTA_dataset.dataset(config,train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        labels = np.loadtxt(config['label_path'], delimiter=',')
    if config['dataset'] == 'MSL':
        train_data = np.load(config['train_data_path'])
        test_data = np.load(config['test_data_path'])
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_val_data = data
        # 划分val
        val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        val_data = data[val_seletcion, :, :]
        train_data = np.delete(data, val_seletcion, axis=0)
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        val_dataset = COUTA_dataset.dataset(config, val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        train_val_dataset = COUTA_dataset.dataset(config, train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.load(config['test_data_path'])
        train_val_data_orig = np.load(config['train_data_path'])
        labels = np.load(config['label_path'])
if config['model'] == 'HSR':
    if config['dataset'] == 'MSL':
        train_data = np.load(config['train_data_path'])
        test_data = np.load(config['test_data_path'])
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_data = data
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = HSR_dataset.dataset(config, train_data, flag='train')
        test_dataset = HSR_dataset.dataset(config, test_data, flag='test')
        # 原始的数据
        test_data_orig = np.load(config['test_data_path'])
        train_val_data_orig = np.load(config['train_data_path'])
        labels = np.load(config['label_path'])
    if config['dataset'] == 'SMAP':
        train_data = np.load(config['train_data_path'])
        test_data = np.load(config['test_data_path'])
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_data = data
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = HSR_dataset.dataset(config, train_data, flag='train')
        test_dataset = HSR_dataset.dataset(config, test_data, flag='test')
        # 原始的数据
        test_data_orig = np.load(config['test_data_path'])
        train_val_data_orig = np.load(config['train_data_path'])
        labels = np.load(config['label_path'])
    if config['dataset'] == 'penism':
        train_data = np.loadtxt(config['train_data_path'], delimiter=',')
        if config['test_data_path'] == 'data/penism/train_data_3.csv':
            test_data = np.loadtxt(config['test_data_path'], delimiter=',')[0:400,:]
        else:
            test_data = np.loadtxt(config['test_data_path'], delimiter=',')[:, :-1]  # 这是如果test是swat_penism的话的话就要注释掉
        # test_data  =np.loadtxt(config['test_data_path'], delimiter=',')#这是如果test是swat_penism的话就使用这句
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step, penism_train_flag=True)  # 处理时间戳，切片
        train_data = data
        train_val_data = data
        # 划分val
        # val_seletcion = random.sample(range(len(data)), int(len(data) * config['val_fac']))
        # val_data = data[val_seletcion, :, :]
        # train_data = np.delete(data, val_seletcion, axis=0)
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = COUTA_dataset.dataset(config, train_data, flag='train')
        # val_dataset = COUTA_dataset.dataset(config,val_data, flag='val')
        test_dataset = COUTA_dataset.dataset(config, test_data, flag='test')
        train_val_dataset = COUTA_dataset.dataset(config, train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        if config['test_data_path'] == 'data/penism/train_data_3.csv':
            test_data_orig = test_data_orig[0:400, :]
            labels = np.zeros(len(test_data_orig))
        else:
            labels = test_data_orig[:, -1]
            test_data_orig = test_data_orig[:, :-1]  # 这是如果test是swat_penism的话的话就要注释掉
    if config['dataset'] == 'SWAT':
        train_data = np.loadtxt(config['train_data_path'],delimiter=',')
        test_data = np.loadtxt(config['test_data_path'],delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_data = data
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = HSR_dataset.dataset(config, train_data, flag='train')
        test_dataset = HSR_dataset.dataset(config, test_data, flag='test')
        train_val_dataset = COUTA_dataset.dataset(config, train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'],delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'],delimiter=',')
        labels = np.loadtxt(config['label_path'],delimiter=',')
    if config['dataset'] == 'SMD':
        train_data = np.loadtxt(config['train_data_path'],delimiter=',')
        test_data = np.loadtxt(config['test_data_path'],delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_data = data
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = HSR_dataset.dataset(config, train_data, flag='train')
        test_dataset = HSR_dataset.dataset(config, test_data, flag='test')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'],delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'],delimiter=',')
        labels = np.loadtxt(config['label_path'])
    if config['dataset'] == 'WADI':
        train_data = np.loadtxt(config['train_data_path'],delimiter=',')
        test_data = np.loadtxt(config['test_data_path'],delimiter=',')
        config['input_dim'] = train_data.shape[-1]
        step = 1
        data = slice_windows_data.process_data(train_data, config, step)  # 处理时间戳，切片
        train_data = data
        train_val_data = train_data
        step = config['step']  # 因为想要覆盖所有的数据
        test_data = slice_windows_data.process_data(test_data, config, step)
        # 制作dataset
        train_dataset = HSR_dataset.dataset(config, train_data, flag='train')
        test_dataset = HSR_dataset.dataset(config, test_data, flag='test')
        train_val_dataset = COUTA_dataset.dataset(config, train_val_data, flag='train_val')
        # 原始的数据
        test_data_orig = np.loadtxt(config['test_data_path'],delimiter=',')
        train_val_data_orig = np.loadtxt(config['train_data_path'],delimiter=',')
        labels = np.loadtxt(config['label_path'],delimiter=',')
if config['model'] == "SSHSR":#注意这个模型前面的20维在dataset里面是模型的输入，后面40维是模型的输出
    if config['dataset'] == 'penism':
        train_data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        test_data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        config['input_dim'] = train_data_orig.shape[-1]
        config['window_size'] = config['base_length']
        labels= test_data_orig[:, -1]
        test_data = test_data_orig[:, :-1]  # 这是如果test是penism的话最后一行是label，所以要去掉
        test_data = slice_windows_data.process_data(test_data, config, step=config['base_length'])
        test_dataset = SSHSR_dataset.dataset(config, test_data)
        test_data_reverse_orig = np.flipud(test_data_orig)[:,:-1]#将test_data_orig倒序
        test_data_reverse = slice_windows_data.process_data(test_data_reverse_orig, config, step=config['base_length'])
        test_reverse_dataset = SSHSR_dataset.dataset(config, test_data_reverse)

        train_data = produce_train_target_data.process_train_data(train_data_orig, config, step=config['train_step'],base_length=config['base_length'],fore_length=config['fore_length'])
        train_target_data = produce_train_target_data.process_target_data(train_data_orig, config,step=config['train_step'],base_length=config['base_length'],fore_length=config['fore_length'])
        train_plus_target = np.concatenate((train_data, train_target_data), axis=1)#将train_data和train_target_data拼接起来

        train_data_reverse_orig = np.flipud(train_data_orig)#将train_data_orig倒序
        train_data_reverse = produce_train_target_data.process_train_data(train_data_reverse_orig, config, step=config['train_step'],base_length=config['base_length'],fore_length=config['fore_length'])
        train_target_data_reverse = produce_train_target_data.process_target_data(train_data_reverse_orig, config,step=config['train_step'],base_length=config['base_length'],fore_length=config['fore_length'])
        train_plus_target_reverse = np.concatenate((train_data_reverse, train_target_data_reverse), axis=1)#将train_data_reverse和train_target_data_reverse拼接起来
        train_plus_target_data = np.concatenate((train_plus_target,train_plus_target_reverse))
        train_plus_target_dataset = SSHSR_dataset.dataset(config, train_plus_target_reverse)
if config['model'] == "GDN":
    if config['dataset'] == 'penism':
        train_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        test_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        train_labels = np.zeros(train_orig.shape[0])
        test_labels = test_orig[:, -1]
        train_input, train_output, train_labels= GDN_proceese_data.process(config,torch.tensor(train_orig).permute(1,0),torch.tensor(train_labels),train=True)
        test_input, test_output, test_labels = GDN_proceese_data.process(config,torch.tensor(test_orig[:, :-1]).permute(1,0), torch.tensor(test_labels))
        train_dataset = GDN_dataset.dataset(config, train_input, train_output, train_labels)
        test_dataset = GDN_dataset.dataset(config, test_input, test_output, test_labels)
    if config['dataset']=='SWAT' or config['dataset']=='SMD':
        train_orig = np.loadtxt(config['train_data_path'], delimiter=',')
        test_orig = np.loadtxt(config['test_data_path'], delimiter=',')
        labels = np.loadtxt(config['label_path'], delimiter=',')
        train_labels = np.zeros(train_orig.shape[0])
        test_labels = labels
        train_input, train_output, train_labels= GDN_proceese_data.process(config,torch.tensor(train_orig).permute(1,0),torch.tensor(train_labels),train=True)
        test_input, test_output, test_labels = GDN_proceese_data.process(config,torch.tensor(test_orig[:, :-1]).permute(1,0), torch.tensor(test_labels))
        train_dataset = GDN_dataset.dataset(config, train_input, train_output, train_labels)
        test_dataset = GDN_dataset.dataset(config, test_input, test_output, test_labels)
    if config['dataset'] == 'MSL':
        train_orig = np.load(config['train_data_path'])
        test_orig = np.load(config['test_data_path'])
        labels = np.load(config['label_path'])
        train_labels = np.zeros(train_orig.shape[0])
        test_labels = labels
        train_input, train_output, train_labels = GDN_proceese_data.process(config,torch.tensor(train_orig).permute(1, 0), torch.tensor(train_labels), train=True)
        test_input, test_output, test_labels = GDN_proceese_data.process(config, torch.tensor(test_orig[:, :-1]).permute(1, 0),torch.tensor(test_labels))
        train_dataset = GDN_dataset.dataset(config, train_input, train_output, train_labels)
        test_dataset = GDN_dataset.dataset(config, test_input, test_output, test_labels)
if config['model']=='MTAD_GAT':
    if config['dataset'] == 'penism':
        train_data = torch.tensor(np.loadtxt(config['train_data_path'], delimiter=','),dtype=torch.float32)
        test_data = torch.tensor(np.loadtxt(config['test_data_path'], delimiter=','),dtype=torch.float32)
        test_labels = test_data[:, -1]
        test_data=test_data[:,:-1]
        test_data_orig = test_data
        train_dataset = MTAD_GAT_dataset.dataset(config, train_data)
        test_dataset = MTAD_GAT_dataset.dataset(config, test_data)
    if config['dataset'] == "MSL":
        train_data = torch.tensor(np.load(config['train_data_path']),dtype=torch.float32)
        test_data = torch.tensor(np.load(config['test_data_path']),dtype=torch.float32)
        test_labels = torch.tensor(np.load(config['label_path']),dtype=torch.float32)
        test_data_orig = test_data
        train_dataset = MTAD_GAT_dataset.dataset(config, train_data)
        test_dataset = MTAD_GAT_dataset.dataset(config, test_data)
    if config['dataset']=='SMD':
        train_data = torch.tensor(np.loadtxt(config['train_data_path'], delimiter=','),dtype=torch.float32)
        test_data = torch.tensor(np.loadtxt(config['test_data_path'], delimiter=','),dtype=torch.float32)
        test_labels = torch.tensor(np.loadtxt(config['label_path'], delimiter=','),dtype=torch.float32)
        test_data_orig = test_data
        train_dataset = MTAD_GAT_dataset.dataset(config, train_data)
        test_dataset = MTAD_GAT_dataset.dataset(config, test_data)
# %%  设定dataloader
if config['model'] == 'TranAD':
    train_dataloader = DataLoader(train_dataset_w, batch_size=config['train_batchsize'],shuffle=False)
    if config['dataset'] =='penism':
        val_dataloader = DataLoader(val_dataset_w, batch_size=len(val_dataset_w), shuffle=False)
    test_dataloader = DataLoader(test_dataset_W, batch_size=len(test_dataset_W), shuffle=False)
if config['model'] == 'AE_basic':
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batchsize'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    train_val_dataloader = DataLoader(train_val_dataset, batch_size=len(train_val_dataset), shuffle=False)
if config['model'] == 'GRELEN':
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batchsize'], shuffle=True)
    config['test_batchsize'] = config['train_batchsize']#设置test_batchsize和train_batchsize都是一样的
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batchsize'], shuffle=False)
if config['model'] == 'COUTA':
    config['train_batchsize'] = config['batch_size']
    config['test_batchsize'] = config['batch_size']
    config['train_val_batchsize'] = config['batch_size']
    config['val_batchsize'] = config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batchsize'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['val_batchsize'], shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_batchsize'], shuffle=False, drop_last=False)
    #train_val_dataloader = DataLoader(train_val_dataset, batch_size=config['train_val_batchsize'], shuffle=False,
    #                                  drop_last=True)
if config['model'] == 'HSR':
    config['train_batchsize'] = config['batch_size']
    config['test_batchsize'] = config['batch_size']
    config['train_val_batchsize'] = config['batch_size']
    config['val_batchsize'] = config['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batchsize'], shuffle=True, drop_last=True)

    if config['dataset'] == 'penism':
        test_dataloader = DataLoader(test_dataset, batch_size=config['test_batchsize'], shuffle=False, drop_last=False)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=config['test_batchsize'], shuffle=False, drop_last=False)
    # train_val_dataloader = DataLoader(train_val_dataset, batch_size=config['train_val_batchsize'], shuffle=False, drop_last=True)
if config['model'] == 'SSHSR':
    batch_size = config['batch_size']
    train_plus_target_dataloader=DataLoader(train_plus_target_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_reverse_dataloader = DataLoader(test_reverse_dataset, batch_size=batch_size, shuffle=False)
if config['model'] == 'GDN':
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
if config['model'] == 'MTAD_GAT':
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
# %% 设定model
if config['model'] == 'TranAD':
    model = TranAD_model.TranAD(config)
    model.to(device)
if config['model'] == 'AE_basic':
    model = AE_basic.AE_basic(config)
    model.to(device)
if config['model'] == 'GRELEN':
    model = GRELEN_model.Grelen(config)
    model.to(device)
if config['model'] == 'COUTA':
    model = COUTA_model.COUTA(config)
    model.to(device)
if config['model'] == 'HSR':
    model = HSR_model.HSR(config)
    model = model.to(device)
    model1 = HSR_model.HSR_1(config)
    model1 = model1.to(device)
if config['model'] == 'SSHSR':
    model = SSHSR_model.SSHSR(config)
    model.to(device)
    model1 = SSHSR_model.SSHSR_1(config)
    model1.to(device)
if config['model'] == 'GDN':
    model = GDN_model.GDN(config)
    model.to(device)
if config['model'] == 'MTAD_GAT':
    model = MTAD_GAT_model.MTAD_GAT(config)
    model.to(device)

# %% 开始训练
if config['model'] == 'TranAD':
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learn_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    l = nn.MSELoss(reduction='none')
    TranAD_trainer.TranADtrainer(config, model, train_dataloader, optimizer, scheduler, l, device)
if config['model'] == 'AE_basic':
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learn_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    l = nn.MSELoss(reduction='none')
    AE_basic_trainer.trainer(config, model, train_dataloader, optimizer, l, device)
if config['model'] == 'GRELEN':
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learn_rate'], weight_decay=1e-5)
    GRELEN_trainer.trainer(config, model, train_dataloader, optimizer, device)
if config['model'] == 'COUTA':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learn_rate'])
    c = COUTA_trainer.trainer(config, model, train_dataloader, val_dataloader, optimizer,device)  # 比较重要的是，这个有一个c后面的tester要用到
    c_copy = c.cpu().detach().numpy()
    if config['dataset'] == 'SWAT':
        np.savetxt('data/SWAT/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/SWAT/c_copy.csv', delimiter=',')
    if config['dataset'] == 'penism':
        np.savetxt('data/penism/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/penism/c_copy.csv', delimiter=',')
    if config['dataset'] == 'WADI':
        np.savetxt('data/WADI/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/WADI/c_copy.csv', delimiter=',')
    if config['dataset'] == 'SMD':
        np.savetxt('data/SMD/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/SMD/c_copy.csv', delimiter=',')
    if config['dataset'] == 'MSL':
        np.savetxt('data/MSL/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/MSL/c_copy.csv', delimiter=',')
    c = torch.tensor(c, dtype=torch.float32).to(device)
if config['model'] == 'HSR':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learn_rate'])
    center = HSR_trainer.trainer(config, model1, model, train_dataloader, optimizer, device)
    c_copy = center.cpu().detach().numpy()
    if config['dataset'] == 'MSL':
        np.savetxt('data/MSL/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/MSL/c_copy.csv', delimiter=',')
    if config['dataset'] == 'SMAP':
        np.savetxt('data/SMAP/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/SMAP/c_copy.csv', delimiter=',')
    if config['dataset'] == 'penism':
        np.savetxt('data/penism/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/penism/c_copy.csv', delimiter=',')
    if config['dataset'] == 'SWAT':
        np.savetxt('data/SWAT/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/SWAT/c_copy.csv', delimiter=',')
    if config['dataset']=='SMD':
        np.savetxt('data/SMD/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/SMD/c_copy.csv', delimiter=',')
    if config['dataset'] == 'WADI':
        np.savetxt('data/WADI/c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/WADI/c_copy.csv', delimiter=',')
    c = torch.tensor(c, dtype=torch.float32).to(device)
if config['model'] == 'SSHSR':
    optimizer= torch.optim.Adam(model.parameters(), lr=config['learn_rate'])
    center = SSHSR_trainer.trainer(config, model1, model, train_plus_target_dataloader, optimizer, device)
    c_copy = center.cpu().detach().numpy()
    if config['dataset'] == 'penism':
        np.savetxt('data/penism/SSHSR_c_copy.csv', c_copy, delimiter=',')
        c = np.loadtxt('data/penism/SSHSR_c_copy.csv', delimiter=',')
if config['model'] == 'GDN':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    GDN_trainer.trainer(config, model, train_dataloader, optimizer)
if config['model'] == 'MTAD_GAT':
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    MTAD_GAT_trainer.trainer(config, model, train_dataloader, optimizer)
# %%开始测试
if config['model'] == 'TranAD':
    fname = 'checkpoints/TranAD_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = TranAD_model.TranAD(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    l = nn.MSELoss(reduction='none')
    torch.zero_grad = True
    model.eval()
    loss, y_pred = TranAD_tester.tranad_tester(config, model, test_data_orig, test_dataloader, l, device, labels=labels,
                                               plot_flag=True, val=False, loss_each_timestamp=True)
    # 这是我自己加的看一下如果是无异常的数据，它的结果会是什么样子
    loss_2 = loss
    if config['eval_method'] =='spot':
        loss_val, y_pred_val = TranAD_tester.tranad_tester(config, model, val_data_orig, val_dataloader, l, device,
                                                       labels=torch.Tensor([0] * 400), train_dataset_flag=True,
                                                       plot_flag=True, val=True, loss_each_timestamp=True)
if config['model'] == 'AE_basic':
    fname = 'checkpoints/AE_basic_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = AE_basic.AE_basic(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    l = nn.MSELoss(reduction='none')
    torch.zero_grad = True
    model.eval()
    loss, y_pred = AE_basic_tester.tester(config, model, test_data_orig, test_dataloader, l, device,
                                          labels=torch.Tensor([0] * 400), plot_flag=True, val=False,
                                          loss_each_timestamp=True)
    # 这是我自己加的看一下如果是无异常的数据，它的结果会是什么样子
    loss_val, y_pred_val = AE_basic_tester.tester(config, model, val_data_orig, val_dataloader, l, device,
                                                  labels=torch.Tensor([0] * 400), plot_flag=True, val=True,
                                                  loss_each_timestamp=True)
if config['model'] == 'GRELEN':
    fname = 'checkpoints/GRELEN_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = GRELEN_model.Grelen(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    loss, outputs = GRELEN_tester.tester(config, model, test_data_orig, test_dataloader, device, plot_flag=True,
                                         val=False, loss_each_timestamp=False)
if config['model'] == 'COUTA':
    fname = 'checkpoints/COUTA_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = COUTA_model.COUTA(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    scores = COUTA_tester.tester(config, model, test_data_orig, test_dataloader, device, c, plot_flag=True, )
if config['model'] == 'HSR':
    fname = 'checkpoints/HSR_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = HSR_model.HSR(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    loss, rep_loss, reps = HSR_tester.tester(config, model, test_data_orig, test_dataloader, device, labels, c,
                                             plot_flag=True)
if config['model'] == 'SSHSR':
    fname = 'checkpoints/SSHSR_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = SSHSR_model.SSHSR(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    final_loss,test_data_fore_final,test_data_recon_final = SSHSR_tester.tester(config, model, test_data_orig[:,:-1], test_dataloader,test_reverse_dataloader, device, labels, c,plot_flag=True)
if config['model'] == 'GDN':
    fname = 'checkpoints/GDN_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = GDN_model.GDN(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    test_loss,test_ground_list, test_predicted_list,test_label_list = GDN_tester.tester(config, model, test_dataloader, plot_flag=True)
if config['model'] == 'MTAD_GAT':
    fname = 'checkpoints/MTAD_GAT_' + config['dataset'] + '/model.ckpt'
    checkpoints = torch.load(fname)
    model = MTAD_GAT_model.MTAD_GAT(config)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    torch.zero_grad = True
    model.eval()
    anomaly_scores,actual,preds,recons = MTAD_GAT_tester.tester(config,model, test_data_orig, test_dataloader, device, plot_flag=True)
# %%计算threshold同时整理预测值，打印最后结果
if config['model'] == 'TranAD':
    if config['eval_method'] == 'best_f1':
        loss = loss.squeeze()
        scores = np.sum(loss, axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(scores) != len(test_data_orig):#因为dataloader的最后有drop_last=True，导致要损失一些数据
            test_data_orig = test_data_orig[:len(scores)]
        scores = scaler.fit_transform(scores.reshape(len(test_data_orig), 1))
        scores_copy = scores
        labels = labels[:len(scores)]
        scores = adjust_scores.adjust_scores(labels, scores)
        eval_info2 = get_metrics.get_best_f1(labels, scores)
        thred = eval_info2[3]
        y_pred = np.where(scores >= thred, 1, 0)
        y_pred_orig = np.where(scores_copy >= thred, 1, 0)
        ploting.prediction_out(y_pred_orig, y_pred.reshape(len(test_data_orig)), labels, config['model'],
                               config['dataset'])
        ploting.loss_eachtimestamp_prediction_out(labels, y_pred.reshape(len(test_data_orig)), loss, config['model'],
                                                  config['dataset'])
        print(eval_info2, 'f1,precision,recall,threshold')
    if config['eval_method'] == 'spot':
        l = nn.MSELoss(reduction='none')
        lossT, _ = TranAD_tester.tranad_tester(config, model, train_val_data_orig, train_val_dataloader, l, device,
                                               train_dataset_flag=True, plot_flag=False)
        lossT = np.squeeze(lossT)
        loss = np.squeeze(loss)
        lossT_final = np.mean(lossT, axis=1)
        loss_final = np.mean(loss, axis=1)
        lt_f, l_f, ls = lossT_final, loss_final, labels  # lt是训练误差，l是测试集误差，ls是label
        y_pred, label, threshold = pot.pot_eval(config, lt_f, l_f, ls)
        score = metric.calc_point2point(y_pred, label)
        # 绘制一下预测的结果：
        # ploting.prediction_out(y_pred,label,config['model'],config['dataset'])
        ploting.loss_eachtimestamp_prediction_out(labels, y_pred, loss_2, config['model'], config['dataset'])

        print(f"f1:{score[0]},\n"
              f"precision:{score[1]},\n"
              f"recall:{score[2]},\n"
              f"ROC/AUC:{score[7]},\n"
              f"threshold:{threshold}")
if config['model'] == 'AE_basic':
    l = nn.MSELoss(reduction='none')
    lossT, _ = AE_basic_tester.tester(config, model, train_val_data_orig, train_val_dataloader, l, device,
                                      plot_flag=False)
    lossT = np.squeeze(lossT)
    loss = np.squeeze(loss)
    lossT_final = np.mean(lossT, axis=1)
    loss_final = np.mean(loss, axis=1)
    lt_f, l_f, ls = lossT_final, loss_final, labels  # lt是训练误差，l是测试集误差，ls是label
    y_pred, label, threshold = pot.pot_eval(config, lt_f, l_f, ls)
    score = metric.calc_point2point(y_pred, label)
    # 绘制一下预测的结果：
    ploting.prediction_out(y_pred, label, config['model'], config['dataset'])

    print(f"f1:{score[0]},\n"
          f"precision:{score[1]},\n"
          f"recall:{score[2]},\n"
          f"ROC/AUC:{score[7]},\n"
          f"threshold:{threshold}")
if config['model'] == 'GRELEN':
    if config['eval_method'] == "best_f1":
        scores = np.sum(loss, axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(scores) != len(test_data_orig):#因为dataloader的最后有drop_last=True，导致要损失一些数据
            test_data_orig = test_data_orig[:len(scores)]
        scores = scaler.fit_transform(scores.reshape(len(test_data_orig), 1))
        scores_copy = scores
        labels = labels[:len(scores)]
        scores = adjust_scores.adjust_scores(labels, scores)
        eval_info2 = get_metrics.get_best_f1(labels, scores)
        thred = eval_info2[3]
        y_pred = np.where(scores >= thred, 1, 0)
        y_pred_orig = np.where(scores_copy >= thred, 1, 0)
        ploting.prediction_out(y_pred_orig, y_pred.reshape(len(test_data_orig)), labels, config['model'],config['dataset'])
        ploting.loss_eachtimestamp_prediction_out(labels, y_pred.reshape(len(test_data_orig)), loss, config['model'],config['dataset'])
        print(eval_info2, 'f1,precision,recall,threshold')
    if config['eval_method']=='spot':

        l = nn.MSELoss(reduction='none')
        lossT, _ = GRELEN_tester.tester(config, model, train_val_data_orig, train_val_dataloader, device, plot_flag=False)
        lossT = np.squeeze(lossT)
        loss = np.squeeze(loss)
        lossT_final = np.mean(lossT, axis=1)
        loss_final = np.mean(loss, axis=1)
        lt_f, l_f, ls = lossT_final, loss_final, labels  # lt是训练误差，l是测试集误差，ls是label
        y_pred, label, threshold = pot.pot_eval(config, lt_f, l_f, ls)
        score = metric.calc_point2point(y_pred, label)
        # 绘制一下预测的结果：
        ploting.prediction_out(y_pred, label, config['model'], config['dataset'])

        print(f"f1:{score[0]},\n"
              f"precision:{score[1]},\n"
              f"recall:{score[2]},\n"
              f"ROC/AUC:{score[7]},\n"
              f"threshold:{threshold}")
if config['model'] == 'COUTA':
    scores_copy = scores
    scores = adjust_scores.adjust_scores(labels, scores)
    eval_info2 = get_metrics.get_best_f1(labels, scores)
    thred = eval_info2[3]
    y_pred = np.where(scores >= thred, 1, 0)
    y_pred_orig = np.where(scores_copy >= thred, 1, 0)
    ploting.prediction_out(y_pred_orig,y_pred, labels, config['model'], config['dataset'])
    print(eval_info2, 'f1,precision,recall,threshold')
if config['model'] == 'HSR':
    if config['eval_method'] == 'best_f1':
        scores = np.sum(loss, axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(scores) != len(test_data_orig):#因为dataloader的最后有drop_last=True，导致要损失一些数据
            test_data_orig = test_data_orig[:len(scores)]
        scores = scaler.fit_transform(scores.reshape(len(test_data_orig), 1))
        rep_loss = scaler.fit_transform(rep_loss.cpu().numpy().reshape(len(test_data_orig), 1))
        scores = scaler.fit_transform(scores + 0.8*rep_loss)
        scores_copy = scores
        labels = labels[:len(scores)]
        scores = adjust_scores.adjust_scores(labels, scores)
        eval_info2 = get_metrics.get_best_f1(labels, scores)
        thred = eval_info2[3]
        y_pred = np.where(scores >= thred, 1, 0)
        y_pred_orig = np.where(scores_copy >= thred, 1, 0)
        ploting.prediction_out(y_pred_orig, y_pred.reshape(len(test_data_orig)), labels, config['model'],
                               config['dataset'])
        ploting.loss_eachtimestamp_prediction_out(labels, y_pred.reshape(len(test_data_orig)), scores_copy, config['model'],
                                                  config['dataset'])
        print(eval_info2, 'f1,precision,recall,threshold')
    if config['eval_method'] == 'spot':
        scoresT = HSR_tester.tester(config, model, val_data_orig, val_dataloader, device, c, plot_flag=True)
        lossT_final = scoresT
        loss_final = scores
        lt_f, l_f, ls = lossT_final, loss_final, labels  # lt是训练误差，l是测试集误差，ls是label
        y_pred, label, threshold = pot.pot_eval(config, lt_f, l_f, ls)
        y_pred_orig = np.where(scores >= threshold, 1, 0)
        score = metric.calc_point2point(y_pred, label)
        # 绘制一下预测的结果：
        ploting.prediction_out(y_pred_orig, y_pred, label, config['model'], config['dataset'])
        # ploting.loss_eachtimestamp_prediction_out(labels, y_pred, loss_2, config['model'], config['dataset'])

        print(f"f1:{score[0]},\n"
              f"precision:{score[1]},\n"
              f"recall:{score[2]},\n"
              f"ROC/AUC:{score[7]},\n"
              f"threshold:{threshold}")
if config['model'] == 'GDN':
    if config['eval_method'] == 'best_f1':
        scores = np.sum(test_loss, axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = scaler.fit_transform(scores.reshape(len(scores), 1))
        scores_copy = scores
        scores = adjust_scores.adjust_scores(test_label_list, scores)
        eval_info2 = get_metrics.get_best_f1(test_label_list, scores)
        thred = eval_info2[3]
        y_pred = np.where(scores >= thred, 1, 0)
        y_pred_orig = np.where(scores_copy >= thred, 1, 0)
        ploting.prediction_out(y_pred_orig, y_pred.reshape(len(test_ground_list)), test_label_list, config['model'],config['dataset'])
        ploting.loss_eachtimestamp_prediction_out(test_label_list, y_pred, scores_copy, config['model'],config['dataset'])
        print(eval_info2, 'f1,precision,recall,threshold')
if config['model']=='MTAD_GAT':
    scores=np.sum(anomaly_scores,axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scores = scaler.fit_transform(scores.reshape(len(scores), 1))
    scores_copy = scores
    if len(actual) != len(test_data_orig):
        test_labels = test_labels[-len(scores):]
    scores = adjust_scores.adjust_scores(test_labels, scores)
    eval_info2 = get_metrics.get_best_f1(test_labels, scores)
    thred = eval_info2[3]
    y_pred = np.where(scores >= thred, 1, 0)
    y_pred_orig = np.where(scores_copy >= thred, 1, 0)
    ploting.prediction_out(y_pred_orig, y_pred.reshape(len(actual)), test_labels, config['model'],config['dataset'])
    ploting.loss_eachtimestamp_prediction_out(test_labels, y_pred.reshape(len(actual)), scores_copy, config['model'],config['dataset'])
    print(eval_info2, 'f1,precision,recall,threshold')
