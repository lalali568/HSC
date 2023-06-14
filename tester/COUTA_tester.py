import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def tester(config, model, test_data_orig, test_dataloader, device,c, plot_flag=True):
    representation_lst =[]
    representation_lst2 = []
    model.eval()
    with torch.no_grad():
        for x in test_dataloader :
            x = x.float().to(device)
            x_output = model(x)
            representation_lst.append(x_output[0])
            representation_lst2.append(x_output[1])
    reps = torch.cat(representation_lst)
    dis = torch.sum((reps - c) ** 2, dim=1).data.cpu().numpy()
    reps_dup = torch.cat(representation_lst2)
    dis2 = torch.sum((reps_dup - c) ** 2, dim=1).data.cpu().numpy()
    dis = dis+dis2
    dis_pad= np.hstack([0*np.ones(test_data_orig.shape[0] - dis.shape[0]),dis])
    score_t=dis_pad

    return score_t