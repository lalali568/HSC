import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import util.Save_Model
from util import ploting

def trainer(config, model, train_dataloader, optimizer):
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    model.train()
    epoch = config['epoch']
    losses = []
    train_start = time.time()
    with tqdm(total=len(train_dataloader) * epoch, unit='batch', leave=True) as pbar:
        for epoch in range(config['epoch']):

            for ii, (x,y) in enumerate(train_dataloader):
                x = x.to(config['device'])
                y = y.to(config['device'])
                optimizer.zero_grad()

                preds, recons = model(x)
                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = torch.sqrt(forecast_criterion(y, preds))
                recon_loss = torch.sqrt(recon_criterion(x, recons))
                loss = forecast_loss + recon_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.update(1)
                pbar.set_description(f"forecast_loss:{forecast_loss},recon_loss:{recon_loss}")


    util.Save_Model.save_model(model, optimizer, config)
    ploting.loss_recored(losses,config['model'],config['dataset'])


