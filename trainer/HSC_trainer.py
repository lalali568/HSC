import torch
import torch.nn as nn
from tqdm import tqdm
import util.Save_Model

def set_c(config, train_dataloader, model, eps=0.01):
    z_ = []
    model.eval()
    with torch.no_grad():
        for x in train_dataloader:
            x = x.float().to(config['device'])
            output = model(x)
            rep =output.reshape(-1,output.shape[-1])
            z_.append(rep.detach())
            if len(z_) ==5000:
                break
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
    model.train()
    loss_hypersphere=[]
    recon_loss = []
    alpha = config['alpha']
    with tqdm(total=len(train_dataloader) * config['epoch'], unit='batch', leave=True) as pbar:
        for epoch in range(config['epoch']):
            loss_hypersphere_epoch = []
            recon_loss_epoch = []
            for ii, x0 in enumerate(train_dataloader):
                x0 = x0.float().to(device)
                optimizer.zero_grad()
                rep,output = model(x0,c)
                loss_model1 = (torch.mean(torch.sum((rep - c) ** 2, dim=1))-config['radius'])**2
                loss_model2 = recon_criterion(output, x0)
                loss_model2 = torch.mean(loss_model2)
                loss = alpha*loss_model1 + loss_model2
                loss.backward()
                optimizer.step()
                loss_hypersphere.append(loss_model1.item())
                recon_loss.append(loss_model2.item())
                loss_hypersphere_epoch.append(loss_model1.item())
                recon_loss_epoch.append(loss_model2.item())
                pbar.update(1)
                pbar.set_description(f"rep_loss_model1:{loss_model1},recon_loss_model2:{loss_model2}")

    util.Save_Model.save_model(model, optimizer, config)
    return c