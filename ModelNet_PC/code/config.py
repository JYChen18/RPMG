import tensorboardX
from os.path import join as pjoin
import configparser

class Parameters():
    def __init__(self):     
        super(Parameters, self).__init__()
   
        
    def read_config(self, fn):
        config = configparser.ConfigParser()
        config.read(fn)
        self.exp_folder= config.get("Record","exp_folder")
        self.data_folder=config.get("Record", "data_folder")
        self.write_weight_folder= pjoin(self.exp_folder, 'weight')
        logdir= pjoin(self.exp_folder, 'log')
        self.logger = tensorboardX.SummaryWriter(logdir)
        
        self.lr =float( config.get("Params", "lr"))
        self.start_iteration=int(config.get("Params","start_iteration"))
        self.total_iteration=int( config.get("Params", "total_iteration"))
        self.save_weight_iteration=int( config.get("Params", "save_weight_iteration"))
        
        self.out_rotation_mode = config.get("Params","out_rotation_mode")

        self.use_rpmg = bool(int(config.get("Params", "use_rpmg")))
        self.rpmg_tau_strategy = int(config.get("Params", "rpmg_tau_strategy"))
        self.rpmg_lambda = float(config.get("Params", "rpmg_lambda"))
        self.sample_num = int(config.get("Params", "sample_num"))
        self.device = int(config.get("Params","device"))
        self.batch = int (config.get("Params","batch"))
                        
    





























