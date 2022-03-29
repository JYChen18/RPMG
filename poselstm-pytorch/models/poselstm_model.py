from tracemalloc import get_traced_memory
from builtins import NotImplementedError
import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pickle
import numpy
import sys 
BASEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(BASEPATH, '..', '..', 'utils'))
import tools
import rpmg


class PoseLSTModel(BaseModel):
    def name(self):
        return 'PoseLSTModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc)

        # load/define networks
        googlenet_weights = None
        if self.isTrain and opt.init_weights != '':
            googlenet_file = open(opt.init_weights, "rb")
            googlenet_weights = pickle.load(googlenet_file, encoding="bytes")
            googlenet_file.close()
            print('initializing the weights from '+ opt.init_weights)
        self.mean_image = np.load(os.path.join(opt.dataroot , 'mean_image.npy'))

        self.netG = networks.define_network(opt.mode, opt.input_nc, opt.lstm_hidden_size, opt.model,
                                      init_from=googlenet_weights, isTest=not self.isTrain,
                                      gpu_ids = self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.sum_criterion = torch.nn.MSELoss(reduction='sum')
            self.mean_criterion = torch.nn.MSELoss(reduction='mean')

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, eps=1,
                                                weight_decay=0.0625,
                                                betas=(self.opt.adambeta1, self.opt.adambeta2))
            self.optimizers.append(self.optimizer_G)
            # for optimizer in self.optimizers:
            #     self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # print('-----------------------------------------------')

    def set_input(self, input, tau):
        input_A = input['A']
        input_B = input['B']
        self.image_paths = input['A_paths']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.tau = tau
        self.gt_r = tools.compute_rotation_matrix_from_quaternion(self.input_B[:, 3:])

    def forward(self):
        self.pred_B = self.netG(self.input_A)
        l = len(self.pred_B) // 2
        loss_weights = [0.3, 0.3, 1]
        for i in range(l):
            out_nd = self.pred_B[2*i+1]
            if 'RPMG' in self.opt.mode:
                out_rmat = rpmg.simple_RPMG.apply(out_nd, self.tau, 0.01, self.opt.beta * loss_weights[i])
            else:
                if(self.opt.mode == "4D_norm"):
                    out_rmat = tools.compute_rotation_matrix_from_quaternion(out_nd) #b*3*3
                elif(self.opt.mode=="6D_GM"):
                    out_rmat = tools.compute_rotation_matrix_from_ortho6d(out_nd) #b*3*3
                elif(self.opt.mode=="9D_SVD"):
                    out_rmat = tools.symmetric_orthogonalization(out_nd)  # b*3*3
                elif (self.opt.mode == "10D"):
                    out_rmat = tools.compute_rotation_matrix_from_10d(out_nd)  # b*3*3
                elif (self.opt.mode == "3D_Euler"):
                    out_rmat = tools.compute_rotation_matrix_from_euler(out_nd)  # b*3*3
                elif (self.opt.mode == "4D_Axis"):
                    out_rmat = tools.compute_rotation_matrix_from_axisAngle(out_nd)  # b*3*3
                else:
                    raise NotImplementedError
            self.pred_B[2*i+1] = out_rmat

    # no backprop gradients
    def test(self):
        self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward(self):
        self.loss_G = 0
        # self.loss_pos = 0
        # self.loss_ori = 0
        loss_weights = [0.3, 0.3, 1]
        for l, w in enumerate(loss_weights):
            mse_pos = self.mean_criterion(self.pred_B[2*l], self.input_B[:, 0:3])
            if 'RPMG' in self.opt.mode:
                mse_ori = self.sum_criterion(self.pred_B[2*l+1], self.gt_r)
                self.loss_G += mse_pos * w + mse_ori
            else:
                mse_ori = self.mean_criterion(self.pred_B[2*l+1], self.gt_r)
                self.loss_G += (mse_pos + mse_ori*self.opt.beta) * w

            # mse_ori = self.sum_criterion(self.pred_B[2*l+1], self.gt_r)

            # self.loss_pos += mse_pos.item() * w
            # self.loss_ori += mse_ori.item() * w * self.opt.beta
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_G.step()

    def get_current_errors(self):
        pos_err = torch.dist(self.pred_B[0], self.input_B[:, 0:3])
        ori_err = tools.compute_geodesic_distance_from_two_matrices(self.gt_r, self.pred_B[1])
        ori_err = ori_err.mean() * 180 / np.pi
        if self.opt.isTrain:
            return OrderedDict([('pos_err', pos_err),
                                ('ori_err', ori_err),
                                ])
        else:
            return [pos_err.item(), ori_err.item()]

    def get_current_pose(self):
        return numpy.concatenate((self.pred_B[0].data[0].cpu().numpy(),
                                  self.pred_B[1].data[0].cpu().numpy()))

    def get_current_visuals(self):
        input_A = util.tensor2im(self.input_A.data)
        # pred_B = util.tensor2im(self.pred_B.data)
        # input_B = util.tensor2im(self.input_B.data)
        return OrderedDict([('input_A', input_A)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
