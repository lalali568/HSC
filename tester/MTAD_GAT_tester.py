import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from util import ploting

def tester(config, model, test_data_orig, test_dataloader, device, plot_flag=True,loss_each_timestamp=True,val=False):
    preds =[]
    recons = []
    forecast_criterion = nn.MSELoss(reduction='none')
    recon_criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for x,y in test_dataloader :
            x = x.float().to(device)
            y = y.float().to(device)
            y_hat,_ = model(x)

            # Shifting input to include the observed value (y) when doing the reconstruction
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = model(recon_x)

            preds.append(y_hat.detach().cpu().numpy())
            # Extract last reconstruction only
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    recons = np.concatenate(recons, axis=0)
    actual = test_data_orig[config['window_size']:].numpy()

    anomaly_scores = np.zeros_like(actual)
    df_dict = {}
    for i in range(preds.shape[1]):
        df_dict[f"Forecast_{i}"] = preds[:, i]
        df_dict[f"Recon_{i}"] = recons[:, i]
        df_dict[f"True_{i}"] = actual[:, i]
        a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + config['gamma'] * np.sqrt(
            (recons[:, i] - actual[:, i]) ** 2)
        anomaly_scores[:, i] = a_score

    if plot_flag:
        ploting.plot_out(actual, recons,anomaly_scores, config['model'], config['dataset'], val=val,caption='for recon part')
        ploting.plot_out(actual, preds, anomaly_scores,config['model'], config['dataset'], val=val,caption='for forecast part')

    return anomaly_scores,actual,preds,recons




