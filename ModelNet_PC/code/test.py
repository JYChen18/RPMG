import torch
import numpy as np
import random
import os
from os.path import join as pjoin
import sys
import argparse

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))
sys.path.insert(0,pjoin(BASEPATH, '..'))
import config as Config
import utils.tools as tools
from model import Model

def test(test_folder, model):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    geodesic_errors_lst = np.array([])
    l = 0
    test_path_list = [os.path.join(test_folder, i) for i in os.listdir(test_folder)]
    for i in range(len(test_path_list)):
        path = test_path_list[i]
        tmp = torch.load(path)
        pc2 = tmp['pc'].cpu().cuda()
        gt_rmat = tmp['rgt'].cpu().cuda()
        out_rmat, out_nd = model(pc2.transpose(1, 2))
        l += ((gt_rmat - out_rmat) ** 2).sum()
        geodesic_errors = np.array(
            tools.compute_geodesic_distance_from_two_matrices(gt_rmat, out_rmat).data.tolist())  # batch
        geodesic_errors = geodesic_errors / np.pi * 180
        geodesic_errors_lst = np.append(geodesic_errors_lst, geodesic_errors)
    l /= len(test_path_list)

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
    np.save(param.write_weight_folder.replace('/weight',''), errors)
    print("Loss: ", l)
    print("median:"+str(np.round(np.median(errors),2)))
    print("avg:" + str(np.round(errors.mean(), 2)))
    print("max:" + str(np.round(errors.max(), 2)))
    print("std:" + str(np.round(np.std(errors), 2)))
    print("1 accuracy:"+str(np.round((errors<1).sum()/len(errors),3)))
    print("3 accuracy:" + str(np.round((errors < 3).sum() / len(errors), 3)))
    print("5 accuracy:"+str(np.round((errors<5).sum()/len(errors),3)))