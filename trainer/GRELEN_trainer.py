import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from time import time
import util.ploting
from util import *

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def trainer(config,model, train_dataloader, optimizer, device):
    prior= np.array(config['prior'])
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    log_prior = log_prior.cuda()
    start=time()
    losses = []
    for epoch in range(config['epoch']):
        kl_train =[]
        nll_train =[]
        for  batch_data in tqdm(train_dataloader, desc='Training epoch ' + str(epoch + 1) + '/' + str(config['epoch'])):
            batch_data=batch_data.to(device)
            encoder_inputs = batch_data
            labels =batch_data[:,:,config['T']-config['target_T']:]
            optimizer.zero_grad()
            prob, output = model(encoder_inputs)
            loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
            loss_nll = nll_gaussian(output, labels, config['variation']).to(device)
            loss = loss_kl + loss_nll
            losses.append(loss.item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            loss.to(device)
            loss.backward()
            optimizer.step()
        nll_train_ = torch.tensor(nll_train)
        kl_train_ = torch.tensor(kl_train)
        print('epoch: %s, kl_train: %.4f, nll_train: %.4f' % (epoch, kl_train_.mean(), nll_train_.mean()))
    total_time = time() - start
    print('Training time: %.2f' % total_time)
    util.ploting.loss_recored(losses,config['model'],config['dataset'])
    util.Save_Model.save_model(model,optimizer,config)



