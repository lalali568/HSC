import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import util.Save_Model
from util import ploting

def set_c(config, train_dataloader, model, eps=0.01):
    z_ = []
    model.eval()
    with torch.no_grad():
        for x in train_dataloader:
            x = x.float().to(config['device'])
            output = model(x)
            rep =output.reshape(-1,output.shape[-1])
            z_.append(rep.detach())
    z_ = torch.cat(z_, dim=0)
    c = torch.mean(z_, dim=0)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

def trainer(config,model1, model, train_dataloader, optimizer, device):
    c = set_c(config, train_dataloader, model1)
    c= c.to(device)
    recon_criterion = nn.MSELoss(reduction='none')
    model1.train()
    loss_hypersphere=[]
    recon_loss = []
    alpha = config['alpha']
    with tqdm(total=len(train_dataloader) * config['epoch'], unit='batch', leave=True) as pbar:
        for epoch in range(config['epoch']):
            loss_hypersphere_epoch = []
            recon_loss_epoch = []
            for ii, x0 in enumerate(train_dataloader):
                model.train()
                x0 = x0.float().to(device)
                optimizer.zero_grad()
                rep,output = model(x0,c)
                out,out_1,out_2 =output
                loss_model1 = (torch.mean(torch.sum((rep - c) ** 2, dim=1))-config['radius'])**2
                loss_out = torch.mean(recon_criterion(out, x0))
                loss_out_1 = torch.mean(recon_criterion(out_1, x0))
                loss_out_2 = torch.mean(recon_criterion(out_2, x0))
                loss = alpha*loss_model1 + loss_out + loss_out_1 + loss_out_2
                loss.backward()
                optimizer.step()
                loss_hypersphere.append(loss_model1.item())
                recon_loss.append(loss_out.item())
                loss_hypersphere_epoch.append(loss_model1.item())
                recon_loss_epoch.append(loss_out.item())
                pbar.update(1)
                pbar.set_description(f"loss_model1:{loss_model1},loss_model2:{loss_out},loss_out_1:{loss_out_1},loss_out_2:{loss_out_2}")

    util.Save_Model.save_model(model, optimizer, config)
    ploting.loss_recored(recon_loss,config['model'],config['dataset'])
    return c