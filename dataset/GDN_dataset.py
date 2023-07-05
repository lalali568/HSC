from torch.utils.data import Dataset
import numpy as np

class dataset(Dataset):
    def __init__(self, config,data_input, data_output, labels):
        self.config = config
        self.data_input = data_input
        self.data_output = data_output
        self.labels = labels

    def __len__(self):
        return len(self.data_input)
    def __getitem__(self,idx):
        return self.data_input[idx], self.data_output[idx], self.labels[idx]
