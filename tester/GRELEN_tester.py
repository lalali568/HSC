import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import ploting

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def process_test_data_orig(test_data_orig,config):
    res= len(test_data_orig) % config['window_size']
    if res != 0:
        test_data_orig=test_data_orig[:-res]
    return test_data_orig

def tester(config, model, test_data_orig, test_dataloader, device, plot_flag=True,val=False,loss_each_timestamp=True):
    #重构的数据
    test_data_orig=process_test_data_orig(test_data_orig,config)
    test_data_recon = []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        encoder_input = batch_data
        prob,output = model(encoder_input)
        test_data_recon.append(output.detach().cpu().numpy())
    #把test_data_recon变成一个numpy数组
    test_data_recon = np.concatenate(test_data_recon, axis=0)
    test_data_recon = test_data_recon.reshape(-1, config['n_node'])
    l= nn.MSELoss(reduction='none')
    loss = l(torch.tensor(test_data_recon), torch.tensor(test_data_orig)).numpy()
    if plot_flag:
        ploting.plot_out(test_data_orig, test_data_recon,loss, config['model'], config['dataset'], val=val)
        if loss_each_timestamp:
            ploting.record_loss(test_data_orig, test_data_recon, config['model'], config['dataset'], val=val)
    return loss,test_data_recon
