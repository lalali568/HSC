
import numpy as np
import torch

from tester import  HSC_tester
from util import Set_Seed, slice_windows_data, get_metrics, adjust_scores
from dataset import HSR_dataset
from torch.utils.data import DataLoader
from models import HSC as HSC_model
from trainer import HSC_trainer
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


with open('config/HSC/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(config[config['dataset']])
    del config[config['dataset']]

Set_Seed.__setseed__(config['random_seed'])
device = config['device']

#process data
if config['dataset'] == 'MSL':
    train_data = np.load(config['train_data_path'])
    test_data = np.load(config['test_data_path'])
    test_data_orig=test_data.copy()
    labels = np.load(config['label_path'])
if config['dataset'] == 'SWAT':
    train_data = np.loadtxt(config['train_data_path'],delimiter=',')
    test_data = np.loadtxt(config['test_data_path'],delimiter=',')
    test_data_orig = test_data.copy()
    labels = np.loadtxt(config['label_path'],delimiter=',')
if config['dataset'] == 'SMD':
    train_data = np.loadtxt(config['train_data_path'],delimiter=',')
    test_data = np.loadtxt(config['test_data_path'],delimiter=',')
    test_data_orig=test_data.copy()
    labels = np.loadtxt(config['label_path'])

config['input_dim'] = train_data.shape[-1]
train_data = slice_windows_data.process_data(train_data, config, config['step'])
test_data = slice_windows_data.process_data(test_data, config, config['window_size'])
train_dataset = HSR_dataset.dataset(config, train_data )
test_dataset = HSR_dataset.dataset(config, test_data)

#dataloader
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)

#model
model = HSC_model.HSC(config)
model = model.to(device)
model1 = HSC_model.HSC_1(config)
model1 = model1.to(device)

# train
optimizer = torch.optim.Adam(model.parameters(), lr=config['learn_rate'])
center = HSC_trainer.trainer(config, model1, model, train_dataloader, optimizer, device)
c_copy = center.cpu().detach().numpy()
if config['dataset'] == 'MSL':
    np.savetxt('data/MSL/c_copy.csv', c_copy, delimiter=',')
    c = np.loadtxt('data/MSL/c_copy.csv', delimiter=',')
if config['dataset'] == 'SWAT':
    np.savetxt('data/SWAT/c_copy.csv', c_copy, delimiter=',')
    c = np.loadtxt('data/SWAT/c_copy.csv', delimiter=',')
if config['dataset']=='SMD':
    np.savetxt('data/SMD/c_copy.csv', c_copy, delimiter=',')
    c = np.loadtxt('data/SMD/c_copy.csv', delimiter=',')
c = torch.tensor(c, dtype=torch.float32).to(device)

# test
fname = 'checkpoints/HSR_' + config['dataset'] + '/model.ckpt'
checkpoints = torch.load(fname)
model = HSC_model.HSC(config)
model.load_state_dict(checkpoints['model_state_dict'])
model.to(device)
torch.zero_grad = True
model.eval()
loss, rep_loss, reps = HSC_tester.tester(config, model, test_data_orig, test_dataloader, device, labels, c)

# result
scores = np.sum(loss, axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
if len(scores) != len(test_data_orig):
    test_data_orig = test_data_orig[:len(scores)]
scores = scaler.fit_transform(scores.reshape(len(test_data_orig), 1))
rep_loss = scaler.fit_transform(rep_loss.cpu().numpy().reshape(len(test_data_orig), 1))
scores = scaler.fit_transform(scores + config['alpha']*rep_loss)
scores_copy = scores
labels = labels[:len(scores)]
auc_value = roc_auc_score(labels, scores)
scores = adjust_scores.adjust_scores(labels, scores_copy)
eval_info2 = get_metrics.get_best_f1(labels, scores)

thred = eval_info2[3]
print(f"precision:{eval_info2[1]} | recall:{eval_info2[2]} | f1:{eval_info2[0]} | auc:{auc_value}")
