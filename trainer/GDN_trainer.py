import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import util.Save_Model
from util import ploting
import time
import torch.nn.functional as F

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    return loss

def trainer(config,model,train_dataloader,optimizer):
    train_loss_list = []
    device = config['device']

    i = 0
    epoch = config['epoch']
    model.train()
    dataloader = train_dataloader
    with tqdm(total=len(train_dataloader) * config['epoch'], unit='batch', leave=True) as pbar:
        for i_epoch in range(epoch):
            auc_loss = 0
            for x,y,attack_labels in dataloader:
                _start = time.time()
                x,y,attack_labels = [item.float().to(device) for item in [x,y,attack_labels]]
                optimizer.zero_grad()
                out = model(x)
                loss = loss_func(out,y)
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                auc_loss += loss.item()
                i += 1
                pbar.update(1)
                pbar.set_description(f'epoch ({i_epoch} / {epoch}) (Loss:{loss})')
            # print(f'epoch ({i_epoch} / {epoch}) (auc_Loss:{auc_loss / len(dataloader)})',flush=True)

    util.Save_Model.save_model(model, optimizer, config)
    ploting.loss_recored(train_loss_list,config['model'],config['dataset'])

