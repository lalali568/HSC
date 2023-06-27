import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import  dense_to_sparse
from torch_geometric.data import Data

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
"""2023-6-24 13:26
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
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),)
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

"""
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 bias=True, dropout=0.2, residual=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                               dilation=dilation)
        self.conv3 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                                 dilation=dilation)

        self.conv3 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, bias=bias,
                               dilation=dilation)
        self.Chomp1d = Chomp1d(padding)  # 这个应该就是为了保证输出的长度和输入的长度一致
        self.dropout = torch.nn.Dropout(dropout)
        self.residual = residual
        self.net = nn.Sequential(self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
                                 self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),

                                )
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
class Hypersphere(nn.Module):
    def __init__(self,config):
        super(Hypersphere,self).__init__()
        self.config = config
        input_dim = config['input_dim']
        hidden_dims = config['hidden_dim']  # 就默认只有一个tcn层吧，试一下
        kernel_size = config['kernel_size']
        tcn_bias = config['tcn_bias']
        dropout = config['dropout']

        self.layers = []
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            dilation_size = 2 ** i
            padding_size = (kernel_size - 1) * dilation_size
            in_channels = input_dim if i == 0 else hidden_dims[i - 1]
            out_channels = hidden_dims[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                          padding=padding_size, dropout=dropout, bias=tcn_bias, residual=True)]

        self.network = nn.Sequential(*self.layers)
        self.l1 = nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True)
        self.l2 = nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True)
        self.act = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = self.l2(self.act(self.l1(out)))
        return out



class HSR_1(nn.Module):#这个是tcn的版本
    def __init__(self,config):
        super(HSR_1, self).__init__()
        self.hypersphere_layer = Hypersphere(config)
    def forward(self, x):
        hypersphere_out = self.hypersphere_layer(x)
        return hypersphere_out



#%%让我试一下用GAT作为后面的decoder
class GATModel(nn.Module):
    def __init__(self,config,in_features,out_features):
        super(GATModel,self).__init__()
        self.config = config
        self.conv = GATConv(in_features, out_features,v2=True, heads=config['num_heads'], dropout=config['dropout'],)
        self.lin = nn.Linear(config['input_dim']*config['num_heads'], config['input_dim'], bias=False)
        self.edge_index = torch.tensor([[i, j] for i in range(config['window_size']) for j in range(config['window_size']) if i != j], dtype=torch.long).t().contiguous()
        self.edge_index = self.edge_index.repeat(config['batch_size'],1).view(2,-1).to(config['device'])
    def forward(self,x):
        x= x.view(-1,x.size(-1))
        x = self.conv(x, self.edge_index)
        x = self.lin(x)
        x=x.view(-1,self.config['window_size'],self.config['input_dim'])
        return x
class HSR_2(nn.Module):
    def __init__(self,config):
        super(HSR_2,self).__init__()
        self.gat1= GATModel(config,config['input_dim'],config['input_dim'])
        self.lin1=nn.Linear(config['input_dim'], config['input_dim'], bias=True)
        self.lin2=nn.Linear(config['input_dim'], config['input_dim'], bias=True)
        self.leaky = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.gat2 = GATModel(config,config['input_dim'],config['input_dim'])
        self.norm = nn.LayerNorm(config['input_dim'])
        self.dropout = nn.Dropout(config['dropout'])
    def forward(self,x):
        out = self.gat1(x)
        out = self.leaky(self.lin1(out))
        out = self.norm(out)
        out = self.dropout(out)
        out = self.gat2(out)
        out = self.leaky(self.lin2(out))
        return out




#%%这是最后总体的框架
class HSR(nn.Module):
    def __init__(self,config):
        super(HSR,self).__init__()
        self.hypersphere_layer = HSR_1(config)
        self.decoder_layer = HSR_2(config)
    def forward(self, x,c):
        hypersphere_out = self.hypersphere_layer(x)
        rep = hypersphere_out
        decoder_out = self.decoder_layer((hypersphere_out)-c)
        return rep,decoder_out



