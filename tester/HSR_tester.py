import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from util import ploting

def tester(config, model, test_data_orig, test_dataloader, device,labels,c, plot_flag=True,loss_each_timestamp=True,val=False):
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
        test_data_orig = torch.tensor(test_data_orig).float().to(device)[:-(len(test_data_orig)-len(test_data_recon))]#去掉最后一些数据，因为zai
    else:
        test_data_orig = torch.tensor(test_data_orig).float().to(device)
    loss = criterion(test_data_recon, test_data_orig)
    reps = test_data_recon.data.cpu().numpy()
    test_data_recon = test_data_recon.data.cpu().numpy()
    test_data_orig = test_data_orig.data.cpu().numpy()
    loss = loss.data.cpu().numpy()
    labels = labels[:len(test_data_recon)]
    if plot_flag:
        ploting.plot_out(test_data_orig, test_data_recon,loss, config['model'], config['dataset'], val=val)
        #if loss_each_timestamp:
            #ploting.record_loss(labels, loss, config['model'], config['dataset'], val=val)

    return loss,rep_loss,reps