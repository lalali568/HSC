from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd


class TranADdataset_W(Dataset):
    def __init__(self, config, flag="train",nosie=False):
        if config['dataset'] == 'penism':
            if flag=="train":
                data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
                data_orig = data_orig[:6000, :]#如果是多次拼接的版本就是[:6000, :]，如果是一个批次的版本就是[:,:]
                #给data_orig加高斯噪声的
                if nosie:
                    noise = np.random.normal(0, 0.5, data_orig.shape)
                    data_orig = data_orig + noise
                data_orig = torch.tensor(data_orig)
                feat=data_orig.shape[1]
                config['feat']=feat
            elif flag == 'val':
                data_orig = np.loadtxt(config['val_data_path'], delimiter=',')
                data_orig = data_orig[:, :-1]
                data_orig = torch.tensor(data_orig)
                config['val_batchsize'] = data_orig.shape[0]
            elif flag == 'test':
                data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
                data_orig = data_orig[:, :-1]
                data_orig = torch.tensor(data_orig)
                config['test_batchsize']=data_orig.shape[0]
            elif flag == 'train_val':
                data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
                data_orig = data_orig[:6000, :]
                data_orig = torch.tensor(data_orig)
                config['train_val_batchsize']=data_orig.shape[0]
            self.data_list = []

            for i, g in enumerate(data_orig):
                if i >= config['window_size']:
                    w = data_orig[i - config['window_size']:i]
                else:
                    w = torch.cat([data_orig[0].repeat(config['window_size'] - i, 1), data_orig[0:i]])
                self.data_list.append(w)

            self.data_list = torch.stack(self.data_list)
        if config['dataset'] == 'MSL':
            if flag=="train":
                data_orig = np.load(config['train_data_path'])
                data_orig = torch.tensor(data_orig)
                feat=data_orig.shape[1]
                config['feat']=feat
            elif flag == 'test':
                data_orig = np.load(config['test_data_path'])
                data_orig = torch.tensor(data_orig)
                config['test_batchsize']=data_orig.shape[0]
            self.data_list = []

            for i, g in enumerate(data_orig):
                if i >= config['window_size']:
                    w = data_orig[i - config['window_size']:i]
                else:
                    w = torch.cat([data_orig[0].repeat(config['window_size'] - i, 1), data_orig[0:i]])
                self.data_list.append(w)
        if config['dataset'] == 'WADI':
            if flag=="train":
                data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
                data_orig = torch.tensor(data_orig)
                feat=data_orig.shape[1]
                config['feat']=feat
            elif flag == 'test':
                data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
                data_orig = torch.tensor(data_orig)
                config['test_batchsize']=data_orig.shape[0]
            self.data_list = []

            for i, g in enumerate(data_orig):
                if i >= config['window_size']:
                    w = data_orig[i - config['window_size']:i]
                else:
                    w = torch.cat([data_orig[0].repeat(config['window_size'] - i, 1), data_orig[0:i]])
                self.data_list.append(w)
        if config['dataset'] == 'SMD':
            if flag=="train":
                data_orig = np.loadtxt(config['train_data_path'], delimiter=',')
                data_orig = torch.tensor(data_orig)
                feat=data_orig.shape[1]
                config['feat']=feat
            elif flag == 'test':
                data_orig = np.loadtxt(config['test_data_path'], delimiter=',')
                data_orig = torch.tensor(data_orig)
                config['test_batchsize']=data_orig.shape[0]
            self.data_list = []

            for i, g in enumerate(data_orig):
                if i >= config['window_size']:
                    w = data_orig[i - config['window_size']:i]
                else:
                    w = torch.cat([data_orig[0].repeat(config['window_size'] - i, 1), data_orig[0:i]])
                self.data_list.append(w)


    def __getitem__(self, item):
        return self.data_list[item]

    def __len__(self):
        return len(self.data_list)



