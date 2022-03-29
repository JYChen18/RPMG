import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from pointnet_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

class PointNet2_seg(nn.Module):
    def __init__(self, out_channel):
        super(PointNet2_seg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_channel, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        return x


class PointNet_cls(nn.Module):
    def __init__(self, out_channel):
        super(PointNet_cls, self).__init__()
        self.feature_extracter = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, out_channel))

    def forward(self, x):
        batch = x.shape[0]
        x = self.feature_extracter(x).view(batch, -1)
        out_data = self.mlp(x)
        return out_data


class PointNet2_cls(nn.Module):
    def __init__(self, out_channel):
        super(PointNet2_cls, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, out_channel))

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        out_data = self.mlp(l3_points.squeeze(-1))
        return out_data


class PointNet_seg(nn.Module):
    def __init__(self, out_channel):
        super(PointNet_seg, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.LeakyReLU()
        )
        self.f2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, out_channel, kernel_size=1)
        )

    def forward(self, x):
        batch = x.shape[0]
        y = self.f1(x)
        z = self.f2(y)
        xx = torch.cat([y,z.repeat(1,1,1024)],1)
        out_data = self.mlp(xx)
        return out_data
