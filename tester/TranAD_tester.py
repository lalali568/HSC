import torch
import numpy as np
import util.ploting

def tranad_tester(config,model,orig_data,dataloader,l,device,labels=None,train_dataset_flag=False,plot_flag=True,val=False,loss_each_timestamp=False):
    bs = len(next(iter(dataloader)))
    for d in dataloader:
        d = d.to(torch.float32)
        d = d.to(device)
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, bs, config['feat'])
        z = model(window, elem)
        if isinstance(z, tuple): z = z[0]
    loss = l(z,elem)
    if plot_flag:
        util.ploting.plot_out(orig_data,z.detach().cpu().numpy()[0],np.squeeze(loss.detach().cpu().numpy()),config['model'],config['dataset'],val=val)
    #if loss_each_timestamp:
        #util.ploting.record_loss(labels,np.squeeze(loss.detach().cpu().numpy()),config['model'],config['dataset'],val=val)
    return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
