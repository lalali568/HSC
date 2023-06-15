from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class dataset(Dataset):
    def __init__(self,config,flag):
        if config['dataset'] == 'SWAT':
            if flag=="train":
                data = np.loadtxt(config['train_data_path'],delimiter=',')
                self.data_list= self.process_train_data(data,config)
            elif flag == 'val':
                print('hh')
            elif flag == 'test':
                data = np.loadtxt(config['test_data_path'], delimiter=',')
                self.data_list = self.process_test_data(data, config)
            elif flag == 'train_val':
                data = np.loadtxt(config['train_data_path'], delimiter=',')
                self.data_list = self.process_test_data(data, config)
        if config['dataset'] == 'penism':
            if flag=="train":
                data = np.loadtxt(config['train_data_path'],delimiter=',')
                self.data_list= self.process_train_data(data,config)
            elif flag == 'val':
                print('hh')
            elif flag == 'test':
                data = np.loadtxt(config['test_data_path'], delimiter=',')
                data = data[:,:-1]
                self.data_list = self.process_test_data(data, config)
            elif flag == 'train_val':
                data = np.loadtxt(config['train_data_path'], delimiter=',')
                data = data[:config['train_val_len'],:]
                self.data_list = self.process_test_data(data, config)
        if config['dataset'] == 'WADI':
            if flag =="train":
                data=np.loadtxt(config['train_data_path'],delimiter=',')
                self.data_list=self.process_train_data(data,config)
            elif flag == 'test':
                data= np.loadtxt(config['test_data_path'], delimiter=',')
                self.data_list = self.process_test_data(data, config)
        if config['dataset'] == 'SMD':
            if flag =="train":
                data=np.loadtxt(config['train_data_path'],delimiter=',')
                self.data_list=self.process_train_data(data,config)
            elif flag == 'test':
                data= np.loadtxt(config['test_data_path'], delimiter=',')
                self.data_list = self.process_test_data(data, config)


    def process_train_data(self,data,config):
        data_list = []
        end_token = config['window_size']
        start_token = 0
        while end_token <= len(data):
            data_list.append(data[start_token:end_token])
            start_token += config['step']
            end_token += config['step']
        data_list=np.array(data_list)
        return data_list

    def process_test_data(self,data,config):
        data_list = []
        end_token = config['window_size']
        start_token = 0
        while end_token <= len(data):
            data_list.append(data[start_token:end_token])
            start_token += config['window_size']
            end_token += config['window_size']
        data_list=np.array(data_list)
        return data_list

    def __getitem__(self, item):
        return self.data_list[item]
    def __len__(self):
        return len(self.data_list)
