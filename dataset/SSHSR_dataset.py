from torch.utils.data import Dataset
import numpy as np
from numpy.random import RandomState

class dataset(Dataset):
    def __init__(self, config, trian_data):
        self.data = trian_data

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        return data


