import  numpy as np
import torch

def process(config,data,labels,train=False):
    x_arr, y_arr = [], []
    labels_arr = []

    slide_win, slide_stride = (config['slide_win'], config['slide_stride'])
    is_train = train

    node_num, total_time_len = data.shape

    rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

    for i in rang:

        ft = data[:, i-slide_win:i]
        tar = data[:, i]

        x_arr.append(ft)
        y_arr.append(tar)

        labels_arr.append(labels[i])

    x = torch.stack(x_arr).contiguous()
    y = torch.stack(y_arr).contiguous()

    labels = torch.Tensor(labels_arr).contiguous()

    return x, y, labels

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()