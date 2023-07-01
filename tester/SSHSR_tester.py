import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from util import ploting

def tester(config, model, test_data_orig, test_dataloader,test_reverse_dataloader, device,labels,c, plot_flag=True,loss_each_timestamp=True,val=False):
    test_data_orig = torch.tensor(test_data_orig).to(device)
    c=torch.tensor(c).to(device)
    criterion = nn.MSELoss(reduction='none')
    model.eval()
    rep_loss_list = []
    recon_part=[]
    fore_part=[]
    with torch.no_grad():
        for x in test_dataloader :
            x = x.float().to(device)
            rep,x_output = model(x,c)
            rep_loss = torch.sum((rep - c) ** 2, dim=-1)
            x_output_recon=x_output.detach()[:,:config['base_length']]
            x_output_fore = x_output.detach()[:, -config['fore_length']:]
            recon_part.append(x_output_recon)
            fore_part.append(x_output_fore)
            rep_loss_list.append(rep_loss.view(-1,1).detach())
    test_data_recon = torch.cat(recon_part).view(-1,config['input_dim'])
    test_data_fore = torch.cat(fore_part).view(-1,config['input_dim'])
    rep_loss = torch.cat(rep_loss_list)

    #下面是针对reverse的
    rep_loss_list_reverse = []
    recon_part_reverse=[]
    fore_part_reverse=[]
    with torch.no_grad():
        for x in test_reverse_dataloader:
            x = x.float().to(device)
            rep,x_output = model(x,c)
            rep_loss_reverse= torch.sum((rep - c) ** 2, dim=-1)
            x_output_recon=x_output.detach()[:,:config['base_length']]
            x_output_fore = x_output.detach()[:, -config['fore_length']:]
            recon_part_reverse.append(x_output_recon)
            fore_part_reverse.append(x_output_fore)
            rep_loss_list_reverse.append(rep_loss_reverse.view(-1,1).detach())
    test_data_recon_reverse = torch.cat(recon_part_reverse).view(-1,config['input_dim']).flip(0)
    test_data_fore_reverse = torch.cat(fore_part_reverse).view(-1,config['input_dim']).flip(0)
    rep_loss = rep_loss+torch.cat(rep_loss_list_reverse)

    test_data_recon_final = (test_data_recon_reverse+test_data_recon)/2

    test_data_fore_final = torch.cat(((test_data_fore_reverse[2*config['base_length']:,:]+test_data_fore[:-2*config['fore_length'],:])/2,test_data_fore[-2*config['base_length']:-config['base_length'], :]),dim=0)
    test_data_fore_final = torch.cat((test_data_fore_final,test_data_fore_reverse[config['fore_length']:2*config['fore_length'], :]),dim=0)
    # if len(test_data_orig)!=len(test_data_recon):
    #     test_data_orig = torch.tensor(test_data_orig).float().to(device)[:-(len(test_data_orig)-len(test_data_recon))]#去掉最后一些数据，因为zai
    # else:
    #     test_data_orig = torch.tensor(test_data_orig).float().to(device)
    recon_loss = criterion(test_data_recon_final, test_data_orig).data.cpu().numpy()
    fore_loss = criterion(test_data_fore_final, test_data_orig).data.cpu().numpy()
    test_data_recon_final = test_data_recon_final.data.cpu().numpy()
    test_data_fore_final = test_data_fore_final.data.cpu().numpy()
    test_data_orig = test_data_orig.data.cpu().numpy()
    final_loss = recon_loss+fore_loss
    if plot_flag:
        ploting.plot_out(test_data_orig, test_data_recon_final,recon_loss, config['model'], config['dataset'],caption="recon plot out")
        ploting.plot_out(test_data_orig, test_data_fore_final,fore_loss, config['model'], config['dataset'],caption="fore plot out")

    return final_loss,test_data_fore_final,test_data_recon_final