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

    def forward(self, x):
        out = self.network(x.transpose(2,1)).transpose(2,1)
        out= self.l2(self.act(self.l1(out)))
        return out

class transformer_layer(nn.Module):
    def __init__(self,config):
        super(transformer_layer,self).__init__()
        self.config = config
        self.attention =  nn.MultiheadAttention(config['input_dim']*config['num_heads'], config['num_heads'], config['dropout'])
        self.Q_map = nn.Linear(config['input_dim'], config['input_dim']*config['num_heads'], bias=True)
        self.K_map = nn.Linear(config['input_dim'], config['input_dim']*config['num_heads'], bias=True)
        self.V_map = nn.Linear(config['input_dim'], config['input_dim']*config['num_heads'], bias=True)
        self.l1 = nn.Linear(config['input_dim']*config['num_heads'], config['input_dim'], bias=True)
        self.layer_norm = nn.LayerNorm(config['input_dim'])
        self.act = torch.nn.LeakyReLU()
        self.dropout = nn.Dropout(config['dropout'])

        #初始化一下参数
        nn.init.xavier_uniform_(self.Q_map.weight)
        nn.init.xavier_uniform_(self.K_map.weight)
        nn.init.xavier_uniform_(self.V_map.weight)
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.constant_(self.Q_map.bias, 0.0)
        nn.init.constant_(self.K_map.bias, 0.0)
        nn.init.constant_(self.V_map.bias, 0.0)
        nn.init.constant_(self.l1.bias, 0.0)

    def forward(self, x):
        out = self.attention(self.Q_map(x),self.K_map(x),self.V_map(x))[0]
        out = self.dropout(self.act(self.l1(out)))
        out = self.layer_norm(out)
        return out


class HSR_1(nn.Module):#这个是tcn的版本
    def __init__(self,config):
        super(HSR_1, self).__init__()
        self.hypersphere_layer = Hypersphere(config)
    def forward(self, x):
        hypersphere_out = self.hypersphere_layer(x)
        return hypersphere_out



"""
class HSR_1(nn.Module):#这个是线性的版本
    def __init__(self,config):
        super(HSR_1, self).__init__()
        self.linear1 = nn.Linear(config['input_dim'], 40, bias=True)
        self.linear2 = nn.Linear(40, 35, bias=True)
        self.linear3 = nn.Linear(35, config['input_dim'], bias=True)
        self.act = torch.nn.LeakyReLU()
        self.drop_out = nn.Dropout(config['dropout'])
    def forward(self, x):
        out = self.linear3(self.drop_out(self.act(self.linear2(self.drop_out(self.act(self.linear1(x)))))))
        return out
"""
"""
class HSR_2(nn.Module):#这个是transformer的版本
    def __init__(self,config):
        super(HSR_2,self).__init__()
        transformer_layers = []
        for i in range(config['num_transformer_layer']):
            transformer_layers.append(transformer_layer(config))
        self.transformer_layers = nn.Sequential(*transformer_layers)
    def forward(self, x):
        transformer_out = self.transformer_layers(x)
        return transformer_out
"""
"""
class HSR_2(nn.Module):
    def __init__(self,config):
        super(HSR_2,self).__init__()
        self.linear1 = nn.Linear(config['input_dim'], 40, bias=True)
        self.linear2 = nn.Linear(40, 35, bias=True)
        #self.linear3 = nn.Linear(35, 51, bias=True)
        #self.linear4 = nn.Linear(51, 40, bias=True)
        #self.linear5 = nn.Linear(40, 35, bias=True)
        self.linear6 = nn.Linear(35, 51, bias=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.activation1 = torch.nn.LeakyReLU()
        self.activation2 = torch.nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear6.weight)
        nn.init.zeros_(self.linear6.bias)
    def forward(self, x):
        out = self.activation1(self.linear1(x))
        out = self.dropout(out)
        out = self.activation1(self.linear2(out))
        out = self.dropout(out)
        #out = self.linear3(out)
       #out = self.activation1(out)
        #out = self.dropout(out)
        #out = self.activation1(self.linear4(out))
        #out = self.dropout(out)
        #out = self.activation1(self.linear5(out))
        #out = self.dropout(out)
        out = self.activation1(self.linear6(out))
        return out
    """

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
