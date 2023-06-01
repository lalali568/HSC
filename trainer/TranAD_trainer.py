from time import time
import numpy as np
import torch
from tqdm import tqdm
import util.Save_Model
from util import ploting


def TranADtrainer(config, model, train_dataloader, optimizer, scheduler, l, device):
    print("----------------start training-------------")
    epoch = config['epoch']
    start = time()
    l1s = []
    accuracy_list = []
    feat = config['feat']
    loss_list = []
    for e in tqdm(list(range(epoch))):
        for i, d in  enumerate(train_dataloader):
            d = d.to(torch.float32)
            d = d.to(device)
            local_bs = d.shape[0]
            window_noise = d.permute(1, 0, 2)
            elem_noise = window_noise[-1, :, :].view(1, local_bs, feat)
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feat)
            z = model(window, elem)#window是windowsize切下来的，elem是整个dataloader里面的完整时序长度

            l1 = (1 / (e + 1)) * l(z[0], elem) - (1 - 1 / (e + 1)) * l(z[0], elem)
            #l1=l(z,elem)
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            loss_list.append(loss.item())  # 把误差值进行记录
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        accuracy_list.append((np.mean(l1s), optimizer.param_groups[0]['lr']))
    ploting.loss_recored(loss_list, config['model'], config['dataset'])
    total_time = time() - start
    print(f"training time: {total_time}s")
    util.Save_Model.save_model(model, optimizer, config,scheduler, accuracy_list)

