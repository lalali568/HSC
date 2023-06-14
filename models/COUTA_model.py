import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 bias=True, dropout=0.2, residual=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                               dilation=dilation)
        self.Chomp1d = Chomp1d(padding)  # 这个应该就是为了保证输出的长度和输入的长度一致
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        if self.residual:
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)
        else:
            return out


class COUTA(nn.Module):
    def __init__(self, config):
        super(COUTA, self).__init__()
        self.config = config
        input_dim = config['input_dim']
        hidden_dims = config['hidden_dim']
        kernel_size = config['kernel_size']
        dropout = config['dropout']
        tcn_bias = config['tcn_bias']

        self.layers = []
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size - 1) * dilation_size
            in_channels = input_dim
            out_channels = hidden_dims[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                          padding=padding_size, dropout=dropout, bias=tcn_bias, residual=True)]

        self.network = nn.Sequential(*self.layers)
        self.l1 = nn.Linear(hidden_dims[-1], config['rep_hidden_dim'],bias=True)
        self.l2 = nn.Linear(config['rep_hidden_dim'], config['emb_dim'],bias=True)
        self.act =torch.nn.LeakyReLU()

        self.l1_dup = nn.Linear(hidden_dims[-1], config['rep_hidden_dim'],bias=True)
        self.pretext_l1 = nn.Linear(hidden_dims[-1], config['pretext_hidden'],bias=True)
        self.pretext_l2 = nn.Linear(config['pretext_hidden'], config['out_dim'], bias=True)



    def forward(self, x):
        out= self.network(x.transpose(2,1)).transpose(2,1)
        out = out[:,-1]
        rep =self.l2(self.act(self.l1(out)))
        score = self.pretext_l2(self.act(self.pretext_l1(out)))
        rep_dup = self.l2(self.act(self.l1_dup(out)))
        return rep,rep_dup,score

