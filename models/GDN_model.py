import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from util import GDN_proceese_data
from torch_geometric.nn.inits import glorot, zeros
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))  # 这个是给每个节点添加一个自环

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        self.node_dim = 0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, config):
        super(GDN, self).__init__()
        self.config = config
        self.node_num =config['node_num']
        self.embed_dim = config['embed_dim']
        self.dim = self.embed_dim
        self.input_dim = config['slide_win']
        self.out_layer_inter_dim = config['out_layer_inter_dim']
        self.out_layer_num = config['out_layer_num']
        self.topk = config['topk']
        self.embedding = nn.Embedding(self.node_num, self.embed_dim)  # 它会生成的矩阵
        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)  # 归一化
        self.gnn_layers = GNNLayer(self.input_dim, self.dim, inter_dim=self.dim + self.embed_dim, heads=1)
        self.node_embedding = None
        self.learned_graph = None
        self.out_layer = OutLayer(self.dim, self.node_num, self.out_layer_num, inter_num=self.out_layer_inter_dim)  # 输出的是线性
        self.cache_edge_index_sets = [None]
        self.cache_embed_index = None
        self.dp = nn.Dropout(config['dropout'])
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data):

        x = data.clone().detach()
        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []

        #根据topk学习图
        all_embeddings =self.embedding(torch.arange(node_num).to(device))
        weight_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(batch_num, 1)
        weights = weight_arr.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
        cos_ji_mat = cos_ji_mat / normed_mat
        topk_num = self.topk
        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        self.learned_graph=topk_indices_ji

        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

        batch_gated_edge_index = GDN_proceese_data.get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
        gcn_out=self.gnn_layers(x, batch_gated_edge_index, all_embeddings, node_num)

        gcn_outs.append(gcn_out)
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out
