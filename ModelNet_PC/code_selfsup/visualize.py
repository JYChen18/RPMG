import matplotlib.pyplot as plt
import torch

def visualize(pc, pred_r, gt_r):
    pc_pred = torch.bmm(pc, pred_r)
    pc_gt = torch.bmm(pc, gt_r)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[...,0], pc[...,1],pc[...,2])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_pred[..., 0], pc_pred[..., 1], pc_pred[..., 2])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_gt[..., 0], pc_gt[..., 1], pc_gt[..., 2])
    plt.savefig('x.png')


