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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.network(x.transpose(2, 1)).transpose(2, 1)
        out = self.l2(self.act(self.l1(out)))
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
#这是HSR_2使用GAT的版本
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
"""

class DRGRU(nn.Module):
    def __init__(self,config):
        super(DRGRU,self).__init__()
        self._activation = torch.tanh
        self.config = config
        self.device = self.config['device']
        self._num_node = self.config['feat_num']
        self._num_units = self.config['GRU_n_dim']
        self._max_diffusion_step = self.config['max_diffusion_step']

        self._gconv_0 = nn.Linear(self._num_units*2*(self._max_diffusion_step+1), self._num_units*2)
        self._gconv_1 = nn.Linear(self._num_units*2*(self._max_diffusion_step+1), self._num_units*2)
        self._gconv_c_0 = nn.Linear(self._num_units*2*(self._max_diffusion_step+1), self._num_units)
        self._gconv_c_1 = nn.Linear(self._num_units*2*(self._max_diffusion_step+1), self._num_units)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self,inputs,hx,adj):
        output_size = 2*self._num_units
        fn = self._gconv
        value =torch.sigmoid(fn(inputs,adj,hx,output_size,bias_start=1.0))
        value = torch.reshape(value,(-1,self._num_node,output_size))
        r,u = torch.split(tensor=value,split_size_or_sections=self._num_units,dim=-1)
        r = torch.reshape(r,(-1,self._num_node*self._num_units))
        u = torch.reshape(u,(-1,self._num_node*self._num_units))

        c=self ._gconv_c(inputs,adj,r*hx,self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u*hx+(1-u)*c
        return new_state

    def _calculate_random_walk0(self,adj_mx,B):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(self.device)
        d = torch.sum(adj_mx, 1)  # 在第二个维度上进行求和
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)  # 把inf替换成0
        d_mat_inv = torch.diag_embed(d_inv)  # 变成对角矩阵
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)  # 这里感觉就是对矩阵进行了归一化，代表随机游走的概率
        return random_walk_mx

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)#感觉这个有点像D0和D1

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_node, -1))
        state = torch.reshape(state, (batch_size, self._num_node, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)


        if self._max_diffusion_step == 0:
            pass
        else:

            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)
            # print('x0_1', x0_1.shape)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)

                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        x1_0 = x1_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_node, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_node, input_size * num_matrices])

        x0_0 = self._gconv_0(x0_0)
        x1_0 = self._gconv_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_node * output_size])

    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_node, -1))
        state = torch.reshape(state, (batch_size, self._num_node, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)
        x1_0 = x1_0.permute(1, 2, 3, 0)

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_node, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_node, input_size * num_matrices])
        x0_0 = self._gconv_c_0(x0_0)
        x1_0 = self._gconv_c_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_node * output_size])


class Graph_learner(nn.Module):
    def __init__(self,config):
        super(Graph_learner,self).__init__()
        self.config = config
        self.graph_head = self.config['graph_head']
        self.mlp1= nn.Linear(config['window_size'], config['window_size'], bias=True)#这一步是把时间维度进行一个映射
        self.Wq= nn.Linear(config['window_size'], config['window_size']*self.graph_head, bias=True)
        self.Wk = nn.Linear(config['window_size'], config['window_size']*self.graph_head, bias=True)
        for m in [self.Wq,self.Wk,self.mlp1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        self.activation = nn.LeakyReLU()
    def forward(self,x):
        B= x.size(0)
        x = self.activation(self.mlp1(x))
        Q = self.Wq(x)
        K = self.Wk(x)
        Q = Q.view(B,self.graph_head,-1,self.config['window_size'])
        K = K.view(B,self.graph_head,-1,self.config['window_size'])
        graph = torch.matmul(Q,K.permute(0,1,3,2))
        graph=graph.sum(dim=(0,1))
        graph = self.guiyihua(graph)
        return graph
    def guiyihua(self,matrix):
        matrix = matrix - torch.min(matrix)
        matrix = matrix / torch.max(matrix)
        return matrix


class EncoderModel(nn.Module):
    def __init__(self,config):
        super(EncoderModel,self).__init__()
        self.config = config
        self.device = config['device']
        self.input_dim = config['GRU_n_dim']
        self.rnn_units =config['GRU_n_dim']
        self.max_diffusion_step = config['max_diffusion_step']
        self.num_nodes = config['feat_num']
        self.num_rnn_layers = config['num_rnn_layers']
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.dcgru_layer = nn.ModuleList([DRGRU(self.config) for _ in range(self.num_rnn_layers)])
        self.linear_out = nn.Linear(self.config['GRU_n_dim'],1)
    def forward(self,input,adj,hidden_state=None):
        batch_size = input.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_rnn_layers,batch_size, self.hidden_state_size).to(self.device)
        hidden_states = []
        output = input
        for layer_num, dcgru_layer in enumerate(self.dcgru_layer):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num],adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)
class HSR_2(nn.Module):
    def __init__(self,config):
        super(HSR_2,self).__init__()
        self.config = config
        self.Graph_learner = Graph_learner(config)
        self.linear_map = nn.Linear(1, config['GRU_n_dim'], bias=True)
        self.encoder_model = EncoderModel(self.config)
        self.num_node = self.config['feat_num']
        self.linear_out = nn.Linear(self.config['GRU_n_dim'],1)
    def forward(self,x,hidden_state=None):
        #首先学习图给下面的图使用
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        adj = self.Graph_learner(x)
        #把大于0.5的部分设为1，否则就是0
        adj[adj>self.config['graph_threshold']]=1
        adj[adj<=self.config['graph_threshold']]=0
        adj= adj.repeat(batch_size,1,1)
        #然后使用DCGRU
        x_projected=self.linear_map(x.unsqueeze(-1))
        x_projected=x_projected.permute(0,1,3,2)
        state_for_output = torch.zeros(x_projected.size()).to(self.config['device'])
        state_for_output[...]=self.encoder(x_projected,adj)

        output = self.linear_out(state_for_output.permute(0,1,3,2)).squeeze(-1)
        output = output.permute(0,2,1)

        return output

    def encoder(self,input,adj):
        encoder_hidden_state=None
        encoder_hidden_state_tensor = torch.zeros(input.shape).to(self.config['device'])
        for t in range(input.shape[-1]):
            _,encoder_hidden_state = self.encoder_model(input[...,t],adj,encoder_hidden_state)
            encoder_hidden_state_tensor[...,t] = encoder_hidden_state[-1,...].reshape(-1,self.num_node,self.config['GRU_n_dim'])
        return encoder_hidden_state_tensor


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



