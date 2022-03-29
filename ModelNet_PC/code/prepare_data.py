'''
from mesh to normalized pc
'''
import numpy as np
import torch
import os
from os.path import join as pjoin
import trimesh
import argparse
import sys
import tqdm 
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0,pjoin(BASEPATH, '../..'))
import utils.tools as tools

def pc_normalize(pc):
    centroid = (np.max(pc, axis=0) + np.min(pc, axis=0)) /2
    pc = pc - centroid
    scale = np.linalg.norm(np.max(pc, axis=0) - np.min(pc, axis=0))
    pc = pc / scale
    return pc, centroid, scale

if __name__ == "__main__": 
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--data_dir", type=str, default='dataset/modelnet40_manually_aligned', help="Path to modelnet dataset")
    arg_parser.add_argument("-c", "--category", type=str, default='airplane', help="category")
    arg_parser.add_argument("-f", "--fix_test", action='store_false', help="for fair comparision")
    args = arg_parser.parse_args()

    sample_num = 4096
    for mode in ['train', 'test']:
        in_folder = pjoin(args.data_dir, args.category, mode)
        out_folder = pjoin(args.data_dir, args.category, mode + '_pc')
        os.makedirs(out_folder, exist_ok=True)


        lst = [i for i in os.listdir(in_folder) if i[-4:] == '.off']
        lst.sort()
        for p in tqdm.tqdm(lst):
            in_path = pjoin(in_folder, p)
            out_path = pjoin(out_folder, p.replace('.off','.pts'))
            if os.path.exists(out_path) and mode == 'train':
                continue
            mesh = trimesh.load(in_path, force='mesh')
            pc, _ = trimesh.sample.sample_surface(mesh, sample_num)
            pc = np.array(pc)
            pc, centroid, scale = pc_normalize(pc) 
            np.savetxt(out_path, pc)
            
            if mode == 'test' and args.fix_test:
                fix_folder = pjoin(args.data_dir, args.category, mode + '_fix')
                os.makedirs(fix_folder, exist_ok=True)
                fix_path = pjoin(fix_folder, p.replace('.off','.pt'))
                pc = np.random.permutation(pc)[:1024,:]
                #each instance sample 10 rotations for test
                rgt = tools.get_sampled_rotation_matrices_by_axisAngle(10).cpu()
                pc = torch.bmm(rgt, torch.Tensor(pc).unsqueeze(0).repeat(10,1,1).transpose(2,1))
                data_dict = {'pc':pc.transpose(1,2), 'rgt':rgt,'centroid':centroid, 'scale':scale}
                torch.save(data_dict, fix_path)
