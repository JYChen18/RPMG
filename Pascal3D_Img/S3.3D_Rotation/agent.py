from utils import TrainClock, KSchedule
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.append(os.path.join(BASEPATH, '..', '..', 'utils'))
import tools
import rpmg
from networks import get_network


def get_agent(config):
    return MyAgent(config)


class MyAgent(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """

    def __init__(self, config):
        self.config = config
        self.clock = TrainClock()
        self.k_schedule = KSchedule(config.k_init, config.k_safe, config.max_iters)

        self.net = get_network(config).cuda()
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        rpmg.logger_init(self.writer)

    def adjust_learning_rate_by_epoch(self, optimizer, cur_epoch, max_epoch):
        """Sets the learning rate to the initial LR decayed by 10 every _N_ epochs"""
        _N_ = max_epoch // 3  # add just learning rate 3 times.
        lr = self.config.lr * (0.1 ** (max(cur_epoch, 0) // _N_))  # 300))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.config.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.config.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.config.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    def forward(self, img,gt):
        img = img.cuda()
        gt = gt.cuda()  # (b, 3, 3)
        pred = self.net(img)  # (b, 9)

        if 'RPMG' in self.config.mode:
            k = self.k_schedule.get_k(self.clock.iteration)
            pred_orth = rpmg.RPMG.apply(pred, k, 0.01, gt, self.clock.iteration)
            loss = self.criterion(pred_orth, gt)
        elif '9D' in self.config.mode:
            pred_orth = tools.symmetric_orthogonalization(pred)
            if self.config.mode == '9D_SVD':
                loss = self.criterion(pred_orth, gt)
            elif self.config.mode == '9D_inf':
                loss = self.criterion(pred, gt.flatten(1))
            else:
                raise NotImplementedError
        elif '6D' in self.config.mode:
            pred_orth = tools.compute_rotation_matrix_from_ortho6d(pred)
            if self.config.mode == '6D_GM':
                loss = self.criterion(pred_orth, gt)
            elif self.config.mode == '6D_inf':
                gt_6d = torch.cat(gt[:, :, 0], gt[:, :, 1], 1)
                loss = self.criterion(pred, gt_6d)
            else:
                raise NotImplementedError
        elif '4D' in self.config.mode:
            pred_orth = tools.compute_rotation_matrix_from_quaternion(pred)
            if self.config.mode == '4D_norm':
                loss = self.criterion(pred_orth, gt)
            elif self.config.mode == '4D_inf':
                gt_q = tools.compute_quaternions_from_rotation_matrices(gt)  # (b, 4)
                loss = self.criterion(pred, gt_q)
            elif self.config.mode == '4D_Axis':
                pred_orth = tools.compute_rotation_matrix_from_axisAngle(pred)
                loss = self.criterion(pred_orth, gt)
            else:
                raise NotImplementedError
        elif self.config.mode == '3D_Euler':
            pred_orth = tools.compute_rotation_matrix_from_euler(pred)
            loss = self.criterion(pred_orth, gt)
        elif self.config.mode == '10D':
            pred_orth = tools.compute_rotation_matrix_from_10d(pred)
            loss = self.criterion(pred_orth, gt)
        else:
            raise NotImplementedError

        err_deg = torch.rad2deg(tools.compute_geodesic_distance_from_two_matrices(pred_orth, gt))  # batch
        return pred, loss, err_deg

    def train_func(self, real_data,syn_data):
        """one step of training"""
        self.net.train()
        img = torch.cat((real_data[0],syn_data[0]),0)
        gt = torch.cat((real_data[1],syn_data[1]),0).squeeze(1)
        pred, loss, err_deg = self.forward(img, gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return pred, loss, err_deg

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()

        with torch.no_grad():
            pred, loss, err_deg = self.forward(data[0],data[1].squeeze(1))

        return pred, loss, err_deg

if __name__ == '__main__':
    max_epoch = 1000
    _N_ = max_epoch // 3  # add just learning rate 3 times.
    for cur_epoch in range(1000):
        lr = 1e-3 * (0.1 ** (max(cur_epoch, 0) // _N_))  # 300))
        if cur_epoch % 10 == 0:
            print(f'epoch {cur_epoch}: {lr}')
