import torch
import numpy as np
import util.ploting

def tester(config,model,orig_data,dataloader,l,device,labels=None,plot_flag=True,val=False,loss_each_timestamp=False):
    bs = len(next(iter(dataloader)))
    for d in dataloader:
        d=d.float()
        d=d.to(device)
        out = model(d)
        out = torch.cat((out[:,0:7],out[:,8:]),dim=1)
    orig_data=torch.tensor(orig_data).float().to(device)
    orig_data=torch.cat((orig_data[:,0:7],orig_data[:,8:]),dim=1)
    loss = l(out,orig_data)
    if plot_flag:
        util.ploting.plot_out(orig_data.detach().cpu().numpy(),out.detach().cpu().numpy(),loss.detach().cpu().numpy(),config['model'],config['dataset'],val=val)
        if loss_each_timestamp:
            util.ploting.record_loss(labels,loss.detach().cpu().numpy(),config['model'],config['dataset'],val=val)
    return loss.detach().cpu().numpy(), out.detach().cpu().numpy()[0]