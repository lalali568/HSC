from time import time
import numpy as np
import torch
from tqdm import tqdm
import util.Save_Model
from util import ploting

def trainer(config, model, train_dataloader, optimizer,l, device):
    print("----------------start training-------------")
    epoch = config['epoch']
    start = time()
    loss_list = []
    with tqdm(total=len(train_dataloader) * config['epoch'], unit='batch', leave=True) as pbar:
        for epoch in range(config['epoch']):
            for data in train_dataloader:
                data = data.float()
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = l(output, data).mean()
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                pbar.update(1)
            pbar.set_description(f"Epoch [{epoch + 1}/{config['epoch']}], Loss: {loss.item():.4f}")
    total_time = time() - start
    print(f"training time: {total_time:.2f} seconds")
    ploting.loss_recored(loss_list, config['model'], config['dataset'])
    total_time = time() - start
    print(f"training time: {total_time}s")
    util.Save_Model.save_model(model, optimizer, config)