import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from util import gumbel_softmax

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear1 = nn.Linear(self.n_in, self.n_hid)
        self.linear2 = nn.Linear(self.n_hid, self.n_out)
        self.bn = nn.BatchNorm1d(self.n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0)*inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self,inputs):
        x = F.elu(self.linear1(inputs))
        x = F.dropout(self.batch_norm(x),training=self.training)
        x = F.elu(self.linear2(x))
        return x


class Graph_learner(nn.Module):
    def __init__(self,config):
        super(Graph_learner,self).__init__()
        self.config=config
        self.head = config['n_head']
        self.n_head_dim = config['n_head_dim']
        self.n_hid = config['graph_learner_n_hid']
        self.n_in = config['T']# 就是有多少个时间点
        self.mlp1 =MLP(self.n_in, self.n_hid, self.n_hid)
        self.Wq = nn.Linear(self.n_hid, self.n_head_dim*self.head)
        self.Wk = nn.Linear(self.n_hid, self.n_head_dim*self.head)
        for m in [self.Wq,self.Wk]:
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)


    def forward(self,inputs):
        #这是对inputs进行映射到潜在空间
        X = self.mlp1(inputs)

        #对其进行attention操作，计算不同点与点之间的关系
        Xq = self.Wq(X)
        Xk = self.Wk(X)
        B,N, n_hid= X.shape
        Xq = Xq.view(B,N, self.head, self.n_head_dim)
        Xk = Xk.view(B,N, self.head, self.n_head_dim)
        Xq = Xq.permute(0,2,1,3)
        Xk = Xk.permute(0,2,1,3)
        probs = torch.matmul(Xq, Xk.permute(0,1,3,2))


        return probs

class DCGRUCell_(nn.Module):
    def __init__(self,device,num_units,max_diffusion_step,num_nodes, nonlinearity='tanh',filter='random',use_gc_for_ru=True):
        super(DCGRUCell_,self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.device = device
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        self._gconv_0 = nn.Linear(self._num_units*2*(self._max_diffusion_step+1), self._num_units*2)
        self._gconv_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_c_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        self._gconv_c_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hx, adj):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj, hx, output_size, bias_start=1.0))

        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv_c(inputs, adj, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    def _calculate_random_walk0(self, adj_mx, B):  # adj_mx是tensor形式
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(self.device)
        d = torch.sum(adj_mx, 1)  # 在第二个维度上进行求和
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)  # 把inf替换成0
        d_mat_inv = torch.diag_embed(d_inv)#变成对角矩阵
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)#这里感觉就是对矩阵进行了归一化，代表随机游走的概率
        return random_walk_mx
    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)#感觉这个有点像D0和D1

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
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
        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        x0_0 = self._gconv_0(x0_0)
        x1_0 = self._gconv_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])

    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
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

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x0_0 = self._gconv_c_0(x0_0)
        x1_0 = self._gconv_c_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])

class EncoderModel(nn.Module):
    def __init__(self,config):
        super(EncoderModel,self).__init__()
        self.config = config
        self.device = config['device']
        self.input_dim = config['GRU_n_dim']
        self.rnn_units = config['GRU_n_dim']
        self.max_diffusion_step = config['max_diffusion_step']
        self.num_nodes = config['n_node']
        self.num_rnn_layers = config['n_rnn_layers']
        self.filter_type = config['filter_type']
        self.hidden_state_size= self.num_nodes * self.rnn_units
        self.dcgru_layers = nn.ModuleList(DCGRUCell_(self.device,self.rnn_units,self.max_diffusion_step,self.num_nodes,\
                                    filter=self.filter_type) for _ in range(self.num_rnn_layers))

    def forward(self, inputs, adj, hidden_state=None):
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)



class Grelen(nn.Module):
    def __init__(self,config):
        super(Grelen, self).__init__()
        self.config=config
        self.len_sequence = config['T']
        self.device = config['device']
        self.GRU_n_dim = config['GRU_n_dim']
        self.graph_learner = Graph_learner(self.config)
        self.num_nodes = config['n_node']
        self.target_T = config['target_T']
        self.linear1 = nn.Linear(1, self.GRU_n_dim)
        nn.init.xavier_normal_(self.linear1.weight.data)
        self.linear1.bias.data.fill_(0.1)
        self.encoder_model = nn.ModuleList([EncoderModel(self.config) for _ in range(self.config['n_head']-1)])
        self.linear_out = nn.Linear(self.GRU_n_dim, 1)
        nn.init.xavier_normal_(self.linear_out.weight.data)
        self.linear_out.bias.data.fill_(0.1)

    def encoder(self,inputs,adj,head):
        encoder_hidden_state=None
        encoder_hidden_state_tensor=torch.zeros(inputs.shape).to(self.config['device'])
        for t in range(self.len_sequence):
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(-1, self.num_nodes,self.GRU_n_dim)
        return encoder_hidden_state_tensor


    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        B=inputs.shape[0]
        #把inputs变成float32的格式
        inputs = inputs.float()

        #这个是把输入映射一下。
        input_projected = self.linear1(inputs.unsqueeze(-1))#在最后增加一个维度，然后再把这个维度扩展到GRU_n_dim
        input_projected  = input_projected.permute(0, 1,3,2)

        #计算prob，这个就是计算节点与节点之间的关系。
        probs = self.graph_learner(inputs)
        mask_loc = torch.eye(self.config['n_node'],dtype=bool).to(self.config['device'])
        probs = probs.masked_select(~mask_loc).view(B,self.config['n_head'],-1).to(self.config['device'])#把构造的图拿出来，把对角线的数据去掉,感觉其实就是把和自己的关系去掉
        prob_reshaped = probs.permute(0,2,1)
        probs=F.softmax(prob_reshaped,dim=-1)
        edges = F.gumbel_softmax(probs, tau = self.config['gumbel_tau'], hard = True).to(self.config['device'])#这就是在采样，属于四种中的哪一种

        adj_list = torch.ones(self.config['n_head'],B,self.config['n_node'],self.config['n_node'],dtype=torch.float32,device=self.config['device'])
        mask = ~torch.eye(self.config['n_node'], dtype=bool).unsqueeze(0).unsqueeze(0).to(self.config['device'])
        mask = mask.repeat(self.config['n_head'],B,1,1).to(self.config['device'])
        adj_list[mask]= edges.permute(2,0,1).flatten()
        state_for_output = torch.zeros(input_projected.shape).to(self.config['device'])
        state_for_output = (state_for_output.unsqueeze(0)).repeat(self.config['n_head'] - 1, 1, 1, 1, 1)
        #现在进入encoder的部分
        for head in range(self.config['n_head']-1):
            state_for_output[head, ...] = self.encoder(input_projected,adj_list[head+1,...],head)

        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)

        output = self.linear_out(state_for_output2).squeeze(-1)[..., self.config['T']-self.config['target_T']:]
        output = output.permute(0, 2, 1)

        return probs, output