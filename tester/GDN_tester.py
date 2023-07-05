import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from util import ploting
import time

def tester(config,model,test_dataloader,plot_flag=True,val=False):
    loss_func = nn.MSELoss(reduction='none')
    device = config['device']
    test_loss_list = []
    now = time.time()

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(test_dataloader)
    model.eval()

    acu_loss = 0
    with torch.no_grad():
        for x, y, labels in test_dataloader:
            x,y,labels = [item.to(device).float() for item in [x,y,labels]]
            predicted = model(x).float().to(device)
            if len(t_test_predicted_list)<=0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
    test_loss_list = loss_func(t_test_predicted_list, t_test_ground_list).data.cpu().numpy()
    test_predicted_list = t_test_predicted_list.data.cpu().numpy()
    test_ground_list = t_test_ground_list.data.cpu().numpy()
    test_labels_list = t_test_labels_list.data.cpu().numpy()
    if plot_flag:
        ploting.plot_out(test_ground_list,test_predicted_list,test_loss_list,config['model'],config['dataset'],val=val)

    return test_loss_list,test_ground_list,test_predicted_list,test_labels_list
