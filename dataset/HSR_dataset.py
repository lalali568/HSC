from torch.utils.data import Dataset
import numpy as np
from numpy.random import RandomState


class dataset(Dataset):
    def __init__(self, config,data,flag='train'):
        if flag == 'train':
            self.data = data
        elif flag == 'test':
            self.data = data
        elif flag == 'train_val':
            self.data = data
        elif flag == 'val':
            self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.array(data, dtype=np.float32)
        return data