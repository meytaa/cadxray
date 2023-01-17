import torch
from torch import optim
import torch.nn as nn
import datetime, os
from build_model import build_model


class compile_nets(object):
    def __init__(self, config):
        # Running device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training parameters
        self.model_type = config['model']['model_name']
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.output_ch = config['model']['output_ch']
        self.tr_strategy= config['training']['tr_strategy']
        self.num_workers = config['training']['num_workers']
        self.gen_method = config['training']['generalization']
        self.aug_prob = config['data']['aug_prob']
        self.init_w = config['model']['init_weights']
        self.loss_fcn = config['training']['loss_fcn']
        self.set_criterion()
        self.lr = config['training']['lr']
        self.lr_decay = config['training']['lr_decay']

        
        # Data assembler
        self.dataset = config['data']['dataset']
        self.imsize=config['data']['image_size']
        self.train_ds = config['train_dataset']
        self.valid_ds = config['valid_dataset']
        self.test_ds = config['test_dataset']
        self.train_loader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size,num_workers = self.num_workers, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_ds, batch_size=self.batch_size,num_workers = self.num_workers, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size,num_workers = self.num_workers, shuffle=False)

        #Hyperparams
        self.hyper_params = {'Model':self.model_type,
                            'Loss Fcn.': self.loss_fcn,
                            'Feat. Extraction LR': self.lr,
                            'Max Epoch No.': self.num_epochs,
                            'Batch Size': self.batch_size,
                            'Aug. Probability:': self.aug_prob,
                            'Initial weights': self.init_w}

        # Storage paths
        current_time = datetime.datetime.now()
        self.run_id = current_time.strftime("%y%m%d-%f")
        self.store_name = 'mdl={mdl}-trnset={ds}-imsize={imsize}-wt={initw8}-lfn={loss}-gn={gen}-lr=({lr},{decay})-ep={ep}-bs={bs}-aug={aug}-id={id}'.format(
                                                        mdl=self.model_type,
                                                        ds=self.dataset,
                                                        imsize=self.imsize,
                                                        initw8=self.init_w,
                                                        loss=self.loss_fcn,
                                                        gen=self.gen_method,
                                                        lr=self.lr,
                                                        decay=self.lr_decay,
                                                        ep=self.num_epochs,
                                                        bs=self.batch_size,
                                                        aug=self.aug_prob,
                                                        id=self.run_id,
                                                        )
        self.short_store_name = self.run_id
        self.layer_dir = os.path.join(config['log']['layer_path'], 'tmp-'+self.store_name)
        self.epoch_layer_dir = os.path.join(config['log']['layer_path'], self.store_name)
        self.model_path = os.path.join(config['log']['model_path'],self.store_name + '.pkl')
        self.result_path = os.path.join(config['log']['result_path'],self.store_name + '.pt')


        if not os.path.exists(config['log']['model_path']): # saving models
            os.makedirs(config['log']['model_path'])
        if not os.path.exists(config['log']['result_path']):# saving results
            os.makedirs(config['log']['result_path'])
        if not os.path.exists(config['log']['layer_path']):# saving f-maps for analyses
            os.makedirs(config['log']['layer_path'])
        if not os.path.exists(self.layer_dir):
            os.makedirs(self.layer_dir)
            os.makedirs(self.epoch_layer_dir)

        # Features
        self.averager_7 = nn.AdaptiveAvgPool2d((7,7))
        self.averager_1 = nn.AvgPool2d((7,7))

        # Building the model
        self.model = build_model(config['model'])
        self.set_optimizer()
        print('so far done!')




    def set_criterion(self):
        if self.loss_fcn=='bce':
            self.criterion_class = nn.BCELoss()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

