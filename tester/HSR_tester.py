import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def tester(config, model, test_data_orig, test_dataloader, device,labels,c):
    criterion = nn.MSELoss(reduction='none')
    recon_representation=[]
    model.eval()
    rep_loss_list = []
    with torch.no_grad():
        for x in test_dataloader :
            x = x.float().to(device)
            rep,x_output = model(x,c)
            rep_loss = torch.sum((rep - c) ** 2, dim=-1)
            recon_representation.append(x_output.detach())
            rep_loss_list.append(rep_loss.view(-1,1).detach())
    test_data_recon = torch.cat(recon_representation).view(-1,config['input_dim'])
    rep_loss = torch.cat(rep_loss_list)
    if len(test_data_orig)!=len(test_data_recon):
        test_data_orig = torch.tensor(test_data_orig).float().to(device)[:-(len(test_data_orig)-len(test_data_recon))]
    else:
        test_data_orig = torch.tensor(test_data_orig).float().to(device)
    loss = criterion(test_data_recon, test_data_orig)
    reps = test_data_recon.data.cpu().numpy()
    loss = loss.data.cpu().numpy()

    return loss,rep_loss,reps