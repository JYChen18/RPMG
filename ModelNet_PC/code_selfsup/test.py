import torch
import numpy as np
import random
import os
from os.path import join as pjoin
import sys
import argparse
from chamfer_distance import ChamferDistance
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))
sys.path.insert(0,pjoin(BASEPATH, '..'))
import config as Config
import utils.tools as tools
from model import Model

def test(test_loader, model, pgt):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    geodesic_errors_lst = np.array([])
    l = 0
    chamfer_loss = ChamferDistance()
    for pc1 in test_loader:
        pc1 = pc1.float().cuda()
        batch, point_num, _ = pc1.shape
        gt_rmat = tools.get_sampled_rotation_matrices_by_axisAngle(batch)#batch*3*3
        gt_rmats = gt_rmat.contiguous().view(batch,1,3,3).expand(batch, point_num, 3,3 ).contiguous().view(-1,3,3)
        pc2 = torch.bmm(gt_rmats, pc1.view(-1,3,1))#(batch*point_num)*3*1
        pc2 = pc2.view(batch, point_num, 3)
        gt_rmat = gt_rmat.float().cuda()
        out_rmat, out_nd = model(pc2.transpose(1, 2))
        gt = pgt.repeat(pc2.shape[0],1,1)
        dist1, dist2 = chamfer_loss(torch.bmm(pc2, out_rmat), gt)
        l += dist2.mean()+dist1.mean()
        geodesic_errors = np.array(
            tools.compute_geodesic_distance_from_two_matrices(gt_rmat, out_rmat).data.tolist())  # batch
        geodesic_errors = geodesic_errors / np.pi * 180
        
        geodesic_errors_lst = np.append(geodesic_errors_lst, geodesic_errors)
    l /= len(test_loader)

    return geodesic_errors_lst, l


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to config")
    arg_parser.add_argument("-c", "--checkpoint", type=int, default=-1, help="checkpoint number")
    args = arg_parser.parse_args()

    param=Config.Parameters()
    param.read_config(pjoin("../configs", args.config))

    test_folder = pjoin(param.data_folder, 'test_fix')
    if args.checkpoint == -1:
        allcp = os.listdir(param.write_weight_folder)
        allcp.sort()
        weight_path = pjoin(param.write_weight_folder, allcp[-1])
    else:
        weight_path = pjoin(param.write_weight_folder, "model_%07d.weight"%args.checkpoint)
    
    with torch.no_grad():
        model = Model(out_rotation_mode=param.out_rotation_mode)
        print("Load " + weight_path)
        f = torch.load(weight_path)
        model.load_state_dict(f['model'])
        model.cuda()
        model.eval()
        errors, l = test(test_folder, model)
    
    print("Loss: ", l)
    print("median:"+str(np.round(np.median(errors),2)))
    print("avg:" + str(np.round(errors.mean(), 2)))
    print("max:" + str(np.round(errors.max(), 2)))
    print("std:" + str(np.round(np.std(errors), 2)))
    print("1 accuracy:"+str(np.round((errors<1).sum()/len(errors),3)))
    print("3 accuracy:" + str(np.round((errors < 3).sum() / len(errors), 3)))
    print("5 accuracy:"+str(np.round((errors<5).sum()/len(errors),3)))