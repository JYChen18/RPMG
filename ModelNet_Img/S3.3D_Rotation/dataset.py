from lib.datasets.Dataset_Base import Dataset_Base
import os
import numpy as np
import torch
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(BASEPATH, '..', '..', 'utils'))
import tools
from torch.utils.data import Dataset, DataLoader
import cv2


class MyDataset(Dataset_Base):
    def __getitem__(self, idx):
        rc = self.recs[idx]
        cate = rc.category
        # img_id = rc.img_id
        quat = rc.so3.quaternion
        quat = torch.from_numpy(quat)

        img = self._get_image(rc)
        img = torch.from_numpy(img)

        sample = dict(idx=idx,
                      label=self.cate2ind[cate],
                      quat=quat,
                      rot_mat=tools.compute_rotation_matrix_from_quaternion(quat[None]).squeeze(),
                      img=img)
        return sample


def get_dataloader(phase, config, sampling=1.):
    dset = MyDataset(config.category, collection=phase, sampling=sampling, net_arch='vgg16')
    dloader = DataLoader(dset, batch_size=config.batch_size, shuffle=phase == 'train', num_workers=config.num_workers)
    return dloader


def data_len():
    cate10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    for cate in cate10:
        print(cate)
        dset = MyDataset(cate, collection='test', sampling=1, net_arch='vgg16')
        print(len(dset))


if __name__ == '__main__':
    from config import get_config

    config = get_config('train')

    dloader = get_dataloader('train', config, sampling=1)
    print(len(dloader))

    sample = next(iter(dloader))

    imgs = sample.get('img').cpu().numpy().transpose((0, 2, 3, 1))
    imgs = imgs * 255 + dloader.dataset.mean_pxl
    imgs = imgs.astype(np.uint8)
    for i, img in enumerate(imgs):
        cv2.imwrite(f'/home/megabeast/test_bingham/data/{i}.png', img)

    print()
