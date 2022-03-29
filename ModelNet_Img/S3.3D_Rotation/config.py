import os
from utils import ensure_dirs
import argparse
import glob
from os.path import join, dirname
from datetime import datetime


def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """

    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        args = self.str2bool_list(args, [
            'no_decay',
        ])

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print(f"{k:20}: {v}")
            self.__setattr__(k, v)

        if 'axis_angle' in self.mode:
            self.num_classes = 4
        elif 'euler' in self.mode:
            self.num_classes = 3
        elif '10D' in self.mode:
            self.num_classes = 10
        else:
            self.num_classes = int(self.mode.split('_')[0][0])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        proj_root = dirname(dirname(os.path.abspath(__file__)))
        print(f'proj root: {proj_root}')
        if self.date is not None:
            time_suffix = self.date
        else:
            if self.is_train and not self.cont:
                # current time
                time_suffix = datetime.now().strftime('%b%d_%H%M%S')
            else:
                time_suffix = sorted(os.listdir(join(proj_root, 'exps', self.exp_name)))[-1]
                print(f'Use the latest time stamp {time_suffix}')
        self.log_dir = join(proj_root, 'exps', self.exp_name, time_suffix, 'log')
        self.model_dir = join(proj_root, 'exps', self.exp_name, time_suffix, 'ckpt')
        ensure_dirs([self.log_dir, self.model_dir])

        # save all the configurations and code
        py_list = glob.glob(join(proj_root, 'S3.3D_Rotation', '**/*.py'), recursive=True)
        log_name = 'log_cont.txt' if self.cont else 'log.txt'
        with open(join(self.log_dir, log_name), 'w') as log:
            for k, v in args.__dict__.items():
                log.write(f'{k:20}: {v}\n')
            log.write('\n\n')
            for py in py_list:
                with open(py, 'r') as f_py:
                    log.write(f'\n*****{f_py.name}*****\n')
                    log.write(f_py.read())
                    log.write('================================================'
                              '===============================================\n')

    def parse(self):
        """initialize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        self._add_basic_config_(parser)
        self._add_dataset_config_(parser)
        self._add_network_config_(parser)
        self._add_training_config_(parser)
        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--exp_name', type=str, default='test')
        group.add_argument('--mode', type=str, default='9D_inf', choices=['9D_SVD', '9D_inf', '9D_RPMG',
                                                                          '6D_GM', '6D_inf', '6D_RPMG',
                                                                          '4D_norm', '4D_inf', '4D_RPMG',
                                                                          'axis_angle', 'euler', '10D','10D_RPMG'])
        group.add_argument('-g', '--gpu_ids', type=str, default=None,
                           help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        group.add_argument('--category', type=str, default='sofa')
        group.add_argument('--batch_size', type=int, default=64, help="batch size")
        group.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        # group.add_argument("--num_classes", type=int, default=9)
        pass

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--max_epoch', type=int, default=600, help="total number of epochs to train")
        group.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        group.add_argument('--no_decay', type=str, default='False')
        group.add_argument('--lr_step_size', type=int, default=400, help="step size for learning rate decay")
        group.add_argument('--k_init', type=float, default=0.05, help="initial k for next goal")
        group.add_argument('--k_safe', type=float, default=0.25, help="max k (safe k) for next goal")
        group.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        group.add_argument('--date', type=str, required=False, help="datetime for model dir")
        group.add_argument('--save_frequency', type=int, default=10, help="save models every x epochs")
        group.add_argument('--val_frequency', type=int, default=1, help="run validation every x epochs")
        group.add_argument('--log_frequency', type=int, default=100, help="visualize output every x iterations")

    @staticmethod
    def str2bool(args, name):
        if getattr(args, name).lower() == 'true':
            setattr(args, name, True)
        else:
            setattr(args, name, False)
        return args

    def str2bool_list(self, args, names):
        for name in names:
            args = self.str2bool(args, name)
        return args


if __name__ == '__main__':
    config = get_config('train')
