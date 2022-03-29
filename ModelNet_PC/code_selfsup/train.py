import torch
import numpy as np
import os
from os.path import join as pjoin
import argparse
import sys 
from chamfer_distance import ChamferDistance

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))
sys.path.insert(0,pjoin(BASEPATH, '..'))
import utils.tools as tools
import utils.rpmg as rpmg
import config as Config
from dataset import SingleInstanceDataset
from model import Model
from test import test

def train_one_iteraton(pc, pgt, param, model, optimizer, iteration, tau):
    optimizer.zero_grad()
    batch=pc.shape[0]
    point_num = param.sample_num

    ###get training data######
    pc1 = torch.autograd.Variable(pc.float().cuda()) #num*3
    gt_rmat = tools.get_sampled_rotation_matrices_by_axisAngle(batch)#batch*3*3
    gt_rmats = gt_rmat.contiguous().view(batch,1,3,3).expand(batch, point_num, 3,3 ).contiguous().view(-1,3,3)
    pc2 = torch.bmm(gt_rmats, pc1.view(-1,3,1))#(batch*point_num)*3*1
    pc2 = pc2.view(batch, point_num, 3) ##batch,p_num,3

    ###network forward########
    out_rmat,out_nd = model(pc2.transpose(1,2))   #output [batch(*sample_num),3,3]

    ####compute loss##########
    if not param.use_rpmg:
        chamfer_loss = ChamferDistance()
        dist1, dist2 = chamfer_loss(torch.bmm(pc2, out_rmat), pgt.repeat(batch,1,1))
        loss = dist1.mean()+dist2.mean()
    else:
        if param.rpmg_tau_strategy == 1:
            out_9d = out_nd.reshape(-1,3,3)
        else:
            out_9d = rpmg.RPMG.apply(out_nd, tau, param.rpmg_lambda, gt_rmat, iteration)
        chamfer_loss = ChamferDistance()
        dist1, dist2 = chamfer_loss(torch.bmm(pc2, out_9d), pgt.repeat(batch,1,1))
        loss = dist1.mean()+dist2.mean()
    loss.backward()
    optimizer.step()

    if iteration % 100 == 0:
        param.logger.add_scalar('train_loss', loss.item(), iteration)
        if param.use_rpmg:
            param.logger.add_scalar('k', tau, iteration)
            param.logger.add_scalar('lambda', param.rpmg_lambda, iteration)
        param.logger.add_scalar('nd_norm', out_nd.norm(dim=1).mean().item(), iteration)

    return loss

        
# pc_lst: [point_num*3]
def train(param):

    torch.cuda.set_device(param.device)
    
    print ("####Initiate model")
    model = Model(out_rotation_mode=param.out_rotation_mode).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr)
    if param.start_iteration != 0:
        read_path = pjoin(param.write_weight_folder, "model_%07d.weight"%param.start_iteration)
        print("Load " + read_path)
        checkpoint = torch.load(read_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = checkpoint['iteration']
    else:
        print('start from beginning')
        start_iteration = param.start_iteration

    print ("start train")
    train_dataset = SingleInstanceDataset(size=800)
    val_dataset = SingleInstanceDataset(size=200)
    pgt = torch.tensor(train_dataset.get_gt()).to(param.device)[None,:].float()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=param.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=param.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    iteration = start_iteration
    while True:
        for data in train_loader:
            model.train()

            #lr decay
            lr = max(param.lr * (0.7 ** (iteration // (param.total_iteration//10))), 1e-5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            iteration += 1
            if param.rpmg_tau_strategy == 1:
                tau = -1
            elif param.rpmg_tau_strategy == 2:
                tau = 2 

            train_loss = train_one_iteraton(data,  pgt, param, model, optimizer, iteration, tau)
            if (iteration % param.save_weight_iteration == 0):
                print("############# Iteration " + str(iteration) + " #####################")
                print('train loss: ' + str(train_loss.item()))

                model.eval()
                with torch.no_grad():
                    angle_list, val_loss = test(val_loader, model, pgt)
                print('val loss: ' + str( val_loss.item()) )
                param.logger.add_scalar('val_loss', val_loss.item(), iteration)
                param.logger.add_scalar('val_median',np.median(angle_list),iteration)
                param.logger.add_scalar('val_mean', angle_list.mean(),iteration)
                param.logger.add_scalar('val_max', angle_list.max(),iteration)
                param.logger.add_scalar('val_5accuracy', (angle_list < 5).sum()/len(angle_list), iteration)
                param.logger.add_scalar('val_3accuracy', (angle_list < 3).sum() / len(angle_list), iteration)
                param.logger.add_scalar('val_1accuracy', (angle_list < 1).sum() / len(angle_list), iteration)
                param.logger.add_scalar('lr', lr, iteration)

                path = pjoin(param.write_weight_folder, "model_%07d.weight"%iteration)
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration}
                torch.save(state, path)

        if iteration >= param.total_iteration:
            break

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to config")
    args = arg_parser.parse_args()

    param=Config.Parameters()
    param.read_config(pjoin("../configs", args.config))

    print(f'use RPMG: {param.use_rpmg}')
    print(f'lambda = {param.rpmg_lambda}')
    if param.rpmg_tau_strategy == 1:
        assert param.out_rotation_mode == 'svd9d'
        print('Tau = Tgt')
    elif param.rpmg_tau_strategy == 2:
        print('Tau = 2')
    rpmg.logger_init(param.logger)
    os.makedirs(param.write_weight_folder, exist_ok=True)

    train(param)



