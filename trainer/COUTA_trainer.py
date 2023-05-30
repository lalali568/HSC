from time import time
import numpy as np
from numpy.random import RandomState
import torch
from tqdm import tqdm
import util.Save_Model
from util import ploting

class DSVDDUncLoss(torch.nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DSVDDUncLoss, self).__init__()
        self.c = c
        self.reduction = reduction
    def forward(self,rep,rep2):
        dis1 = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep2 - self.c) ** 2, dim=1)
        var = (dis1 - dis2) ** 2

        loss = 0.5 * torch.exp(torch.mul(-1, var)) * (dis1 + dis2) + 0.5 * var

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def set_c(config,train_dataloader,model,eps=0.1):
    z_ = []
    model.eval()
    with torch.no_grad():
        for x in train_dataloader:
            x = x.float().to(config['device'])
            output = model(x)
            rep =output[0]
            z_.append(rep.detach())
    z_ = torch.cat(z_, dim=0)
    c = torch.mean(z_, dim=0)
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

def create_batch_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False):#这个是用来创建负样本的，就是对数据进行变形
    """
    create a batch of negative samples based on the input sequences,
    the output batch size is the same as the input batch size
    :param batch_seqs: input sequences
    :param max_cut_ratio:
    :param seed:
    :param return_mul_label:
    :param type:
    :param ss_type:
    :return:

    """
    rng = np.random.RandomState(seed=seed)

    batch_size, l, dim = batch_seqs.shape
    cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)#它这个好像就是限制在后面的百分之50
    n_cut_dim = rng.randint(1, dim+1, size=batch_size)
    cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

    if type(batch_seqs) == np.ndarray:
        batch_neg = batch_seqs.copy()
        neg_labels = np.zeros(batch_size, dtype=int)
    else:
        batch_neg = batch_seqs.clone()
        neg_labels = torch.LongTensor(batch_size)

    flags = rng.randint(1e+5, size=batch_size)

    n_types = 6
    for ii in range(batch_size):
        flag = flags[ii]

        # collective anomalies
        if flag % n_types == 0:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
            neg_labels[ii] = 1

        elif flag % n_types == 1:
            batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
            neg_labels[ii] = 1

        # contextual anomalies
        elif flag % n_types == 2:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
            neg_labels[ii] = 2

        elif flag % n_types == 3:
            mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
            batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
            neg_labels[ii] = 2

        # point anomalies
        elif flag % n_types == 4:
            batch_neg[ii, -1, cut_dim[ii]] = 2
            neg_labels[ii] = 3

        elif flag % n_types == 5:
            batch_neg[ii, -1, cut_dim[ii]] = -2
            neg_labels[ii] = 3

    if return_mul_label:
        return batch_neg, neg_labels
    else:
        neg_labels = torch.ones(batch_size).long()
        return batch_neg, neg_labels

def trainer(config, model, train_dataloader,val_dataloader, optimizer, device):
    c= set_c(config,train_dataloader,model)
    criterion_oc_umc = DSVDDUncLoss(c=c,reduction='mean')
    criterion_mse = torch.nn.MSELoss(reduction='mean')
    #early_stp = EarlyStopping(intermediate_dir=self.model_dir,patience=7, delta=1e-6, model_name='couta', verbose=False)   #early stop暂时先不看吧
    y0 = -1 * torch.ones(config['batch_size']).float().to(config['device'])

    model.train()
    #pbar = tqdm(range(config['epoch']*len(train_dataloader)),dynamic_ncols=True, smoothing=0.01)
    with tqdm(total=len(train_dataloader) * config['epoch'], unit='batch', leave=True) as pbar:
        for i in range(config['epoch']):
            rng = RandomState(seed=config['random_seed'])
            epoch_seed = rng.randint(0, 1e+6, len(train_dataloader))
            loss_lst, loss_oc_lst, loss_ssl_lst = [], [], []
            for ii, x0 in enumerate(train_dataloader):
                x0 = x0.float().to(config['device'])
                x0_output = model(x0)
                rep = x0_output[0]
                rep2 = x0_output[1]
                loss_oc = criterion_oc_umc(rep, rep2)
                # nac的阶段了
                neg_cand_idx = RandomState(epoch_seed[ii]).randint(0, config['batch_size'], config['neg_batch_size'])
                x1, y1 = create_batch_neg(batch_seqs=x0[neg_cand_idx], max_cut_ratio=config['max_cut_ratio'],
                                          seed=epoch_seed[ii], return_mul_label=False)
                x1 = x1.float().to(config['device'])
                y1 = y1.float().to(config['device'])
                y = torch.hstack([y0, y1])
                x1_output = model(x1)
                pred_x1 = x1_output[-1]
                pred_x0 = x0_output[-1]

                out = torch.cat([pred_x0, pred_x1]).view(-1)  # 拼接在一起然后再展开成一维数组
                loss_ssl = criterion_mse(out, y)
                loss = loss_oc + config['alpha'] * loss_ssl
                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss_lst.append(loss)
                loss_oc_lst.append(loss_oc)
                loss_ssl_lst.append(loss_ssl)
                pbar.update(1)
            pbar.set_description(f"Epoch {i + 1}/{config['epoch']} - Loss: {loss.item():.6f}, Loss_oc: {loss_oc.item():.6f}, Loss_ssl: {loss_ssl.item():.6f}")


        epoch_loss = torch.mean(torch.stack(loss_lst)).item()
        epoch_loss_oc = torch.mean(torch.stack(loss_oc_lst)).item()
        epoch_loss_ssl = torch.mean(torch.stack(loss_ssl_lst)).item()

        val_loss = np.NAN
        if val_dataloader is not None:
            val_loss = []
            with torch.no_grad():
                for x0 in val_dataloader:
                    x0 = x0.float().to(config['device'])
                    x0_output = model(x0)
                    rep = x0_output[0]
                    rep2 = x0_output[1]
                    loss_oc = criterion_oc_umc(rep,rep2)
                    val_loss.append(loss_oc)
                val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

        if (i + 1) % 10 == 0:
            print(f'epoch:{i+1},train_loss:{epoch_loss},train_loss_oc:{epoch_loss_oc},train_loss_ssl:{epoch_loss_ssl},val_loss:{val_loss}')

    util.Save_Model.save_model(model, optimizer, config)
    return c



