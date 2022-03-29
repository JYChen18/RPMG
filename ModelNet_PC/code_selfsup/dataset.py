import torch
import numpy as np
import trimesh

def pc_normalize(pc):
    centroid = (np.max(pc, axis=0) + np.min(pc, axis=0)) /2
    pc = pc - centroid
    scale = np.linalg.norm(np.max(pc, axis=0) - np.min(pc, axis=0))
    pc = pc / scale
    return pc, centroid, scale
    
class SingleInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, sample_num=1024, size=800):
        super(SingleInstanceDataset, self).__init__()
        pgt =  trimesh.load('chair_0003.obj')
        self.pgt, _, _ = pc_normalize(np.array(pgt.vertices))
        self.sample_num = sample_num
        self.size = size
        print(f"use single instance! dataset size: {self.size}")

    def __getitem__(self, index):
        pc = np.random.permutation(self.pgt)
        return pc[:self.sample_num, :].astype(float)

    def __len__(self):
        return self.size
    
    def get_gt(self):
        return self.pgt[:self.sample_num]