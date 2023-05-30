import torch
from torch.nn import TransformerDecoder,TransformerEncoder
import torch.nn as nn

class AE_basic(nn.Module):
    def __init__(self,config):
        super(AE_basic, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config['data_dim'], config['hidden_dim']),
            nn.Tanh(),
            nn.Linear(config['hidden_dim'], config['latent_dim']))
        self.decoder = nn.Sequential(
            nn.Linear(config['latent_dim'], config['hidden_dim']),
            nn.Tanh(),
            nn.Linear(config['hidden_dim'], config['data_dim']),
            nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x