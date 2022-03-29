import torch
import torch.nn as nn
import sys
import os
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))
import utils.tools as tools
from pointnets import PointNet2_cls

class Model(nn.Module):
    def __init__(self, out_rotation_mode="Quaternion"):
        super(Model, self).__init__()
        
        self.out_rotation_mode = out_rotation_mode
        
        if(out_rotation_mode == "Quaternion"):
            self.out_channel = 4
        elif (out_rotation_mode  == "ortho6d"):
            self.out_channel = 6
        elif (out_rotation_mode  == "svd9d"):
            self.out_channel = 9
        elif (out_rotation_mode  == "10d"):
            self.out_channel = 10
        elif out_rotation_mode == 'euler':
            self.out_channel = 3
        elif out_rotation_mode == 'axisangle':
            self.out_channel = 4
        else:
            raise NotImplementedError

        print(out_rotation_mode)

        self.model = PointNet2_cls(self.out_channel)
        

    #pt b*point_num*3
    def forward(self, input):
        out_nd = self.model(input)

        if(self.out_rotation_mode == "Quaternion"):
            out_rmat = tools.compute_rotation_matrix_from_quaternion(out_nd) #b*3*3
        elif(self.out_rotation_mode=="ortho6d"):
            out_rmat = tools.compute_rotation_matrix_from_ortho6d(out_nd) #b*3*3
        elif(self.out_rotation_mode=="svd9d"):
            out_rmat = tools.symmetric_orthogonalization(out_nd)  # b*3*3
        elif (self.out_rotation_mode == "10d"):
            out_rmat = tools.compute_rotation_matrix_from_10d(out_nd)  # b*3*3
        elif (self.out_rotation_mode == "euler"):
            out_rmat = tools.compute_rotation_matrix_from_euler(out_nd)  # b*3*3
        elif (self.out_rotation_mode == "axisangle"):
            out_rmat = tools.compute_rotation_matrix_from_axisAngle(out_nd)  # b*3*3

        return out_rmat, out_nd


        
        
        
        
        
        
        
        
    
