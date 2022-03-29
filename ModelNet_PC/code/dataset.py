import torch
import os
import numpy as np

class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder,sample_num=1024):
        super(ModelNetDataset, self).__init__()
        self.paths = [os.path.join(data_folder, i) for i in os.listdir(data_folder)]
        self.sample_num = sample_num
        self.size = len(self.paths)
        print(f"dataset size: {self.size}")

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        pc = np.loadtxt(fpath)
        pc = np.random.permutation(pc)
        return pc[:self.sample_num, :].astype(float)

    def __len__(self):
        return self.size