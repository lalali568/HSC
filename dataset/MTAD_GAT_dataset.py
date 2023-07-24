from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self, config,data):
        self.data = data
        self.window = config['window_size']
        self.horizon = config['horizon']

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window