import torch
from torch import optim
import torch.nn as nn
import datetime, time, os, math
from build_model import build_model
from utility import AverageMeter, Appender, get_AUROC
import numpy as np
import neptune.new as neptune
import json

class compile_nets(object):
    def __init__(self, config):
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
        self.validation_flag = config['training']['validation_flag']
        self.set_criterion()
        self.lr = float(config['training']['lr'])
        self.lr_decay = float(config['training']['lr_decay'])
        self.machine = config['data']['machine']
        
        # Data assembler
        self.train_dataset = config['data']['train_dataset']
        self.test_dataset = config['data']['test_dataset']
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
        self.store_name = 'mdl={mdl}-dataset=({trds}-{tsds})-imsize={imsize}-wt={initw8}-lfn={loss}-gn={gen}-lr=({lr},{decay})-ep={ep}-bs={bs}-aug={aug}-id={id}'.format(
                        mdl=self.model_type,
                        trds=self.train_dataset,
                        tsds=self.test_dataset,
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
        self.model_path = os.path.join(config['log']['model_path'],self.store_name + '.pt')
        self.result_path = os.path.join(config['log']['result_path'],self.store_name + '.json')


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
        model_creator = build_model(config['model'])
        if config['training']['load_model_id'] and config['model']['init_weights']=='pretrained':
            try:
                pretrained_model_name = model_creator.load_pretrain(model_id = config['training']['load_model_id'])
                print('Pretraied model loaded\nModel ID: {}\nModel path: {}'.format(config['training']['load_model_id'], pretrained_model_name))
            except:
                print('Pretrained model with ID={} does not exist or not matched to the model!.'.format(config['training']['load_model_id']))
        self.model = model_creator.model
        self.set_optimizer()
        model_creator.display()
        self.set_device()
        print('so far done!')

        self.run = neptune.init_run(
                project="meyta/cadxray-generalization",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGRmYzUzNS0yZGY3LTRmNWEtYWYwNi03YzM2MWU0MmM5YzMifQ==",
            )
        self.set_neptune_config()
    #======================================= Methods ===========================================#
    #===========================================================================================#

    def set_criterion(self):
        if self.loss_fcn=='bce':
            self.criterion_class = nn.BCELoss()
        print('Loss function is set to {}.'.format(self.criterion_class))

    def set_optimizer(self):
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.model.classifier.parameters(), self.lr)

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.model.zero_grad()

    # Running device
    def set_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print('Cool! {} GPUs are used for training'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        return 

    def set_neptune_config(self):
        self.run["config/model"] = type(self.model).__name__
        self.run["config/criterion"] = type(self.criterion_class).__name__
        self.run["config/optimizer"] = type(self.optimizer).__name__
        self.run["config/initial weights"] = self.init_w
        self.run["config/num epochs"] = self.num_epochs
        self.run["config/batch size"] = self.batch_size
        self.run["config/lr"] = self.lr
        self.run["config/lr decay"] = self.lr_decay
        self.run["config/train strategy"] = self.tr_strategy
        self.run["config/generalization"] = self.gen_method
        self.run["config/machine"] = self.machine
        self.run["config/train dataset"] = self.train_dataset
        self.run["config/test dataset"] = self.test_dataset
        self.run['config/store name'] = self.store_name


    #====================================== Training ===========================================#
    #===========================================================================================#

    def train(self):
        # start_time = time.time()
        run_status = Appender()
        best_val_loss = 500
        train_epoch_losses = AverageMeter()
        valid_epoch_losses = AverageMeter()
        train_iteration_total = math.ceil(len(self.train_loader.dataset)/self.batch_size)
        valid_iteration_total = math.ceil(len(self.valid_loader.dataset)/self.batch_size)
        train_iteration_num= 0
        val_itr_c= 0
        run_status.update('training started from {}'.format(self.init_w))
        auroc_train = Appender()
        auroc_valid = Appender()
        for epoch in range(self.num_epochs):
            self.model.train()
            d_label = Appender()
            d_pred = Appender()
            epoch_loss = AverageMeter()
            print('Training')
            start_time = time.time()
            for i, (image, label, _) in enumerate (self.train_loader):
                image = image.to(self.device)
                label = label.float().to(self.device)
                pred = torch.sigmoid(self.model(image))

                loss = self.criterion_class(pred,label)
                d_label.update(label.detach().cpu().numpy())
                d_pred.update(pred.detach().cpu().numpy())
                self.reset_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss.update(loss.item(),image.size(0))
                # self.run('train/batch loss').append(loss.item())
                end_time = time.time()
                print('Epoch [ %4d/%d ] - Iteration [ %5d / %d ] --> Loss = %.4f - iteration time:%.4f '%(epoch+1, self.num_epochs, i+1,train_iteration_total,loss.item(), end_time-start_time))
                start_time = end_time
                train_iteration_num +=1


            train_epoch_losses.update(epoch_loss.avg)
            # epoch_loss /=len(self.train_loader.dataset)
            d_pred = np.concatenate(d_pred.elements)
            d_label = np.concatenate(d_label.elements)
            auroc_train_epoch = get_AUROC(d_label,d_pred)
            auroc_train_epoch_mean = np.array(auroc_train_epoch).mean()
            self.run["train/epoch/AUROC - mean"].append(auroc_train_epoch_mean)            
            auroc_train.update(auroc_train_epoch)
            for i, pathology in enumerate(self.train_ds.pathologies):
                self.run["train/epoch/AUROC - {}".format(pathology)].append(auroc_train_epoch[i])
            self.run["train/epoch/loss"].append(epoch_loss.avg)
            if self.validation_flag:
                self.model.eval()
                d_val_label = Appender()
                d_val_pred = Appender()
                epoch_val_loss = AverageMeter()
                print('Validation')
                with torch.no_grad():
                    for i, (image, label,_) in enumerate (self.valid_loader):
                        image = image.to(self.device)
                        label = label.to(self.device).float()
                        pred = torch.sigmoid(self.model(image))
                        loss = self.criterion_class(pred,label)
                        d_val_label.update(label.detach().cpu().numpy())
                        d_val_pred.update(pred.detach().cpu().numpy())
                        epoch_val_loss.update(loss.item(), image.size(0))
                        # self.run('valid/batch loss').log(loss.item())
                        print('Epoch [ %4d/%d ] - Iteration [ %5d/%d ] --> Validation loss = %.4f'%(epoch+1, self.num_epochs, 1+i,valid_iteration_total, loss.item()))
                        val_itr_c +=1

                valid_epoch_losses.update(epoch_val_loss.avg)
                d_val_pred = np.concatenate(d_val_pred.elements)
                d_val_label = np.concatenate(d_val_label.elements)
                auroc_valid_epoch = get_AUROC(d_val_label,d_val_pred)
                auroc_valid_epoch_mean = np.array(auroc_valid_epoch).mean()
                self.run["valid/epoch/AUROC - mean"].log(auroc_valid_epoch_mean)
                auroc_valid.update(auroc_valid_epoch)
                for i, pathology in enumerate(self.valid_ds.pathologies):
                    self.run["valid/epoch/AUROC - {}".format(pathology)].append(auroc_valid_epoch[i])
                self.run["valid/epoch/loss"].append(epoch_val_loss.avg)
                print('Epoch: %4d/%d --> Train Loss: %.4f  Validation Loss: %.4f'%(epoch+1, self.num_epochs, epoch_loss.val, epoch_val_loss.val))

				# Save Best model
                if epoch_val_loss.avg <= best_val_loss:
                    best_val_loss = epoch_val_loss.avg
                    best_epoch = epoch
                    best_model = self.model
                    best_auroc_valid = auroc_valid_epoch
                    print('Best %s model score is reached now: %.4f'%(self.model_type,best_val_loss))
                    model_scripted = torch.jit.script(self.model)
                    model_scripted.save(self.model_path)
                    # model = torch.jit.load(self.model_path)
                    self.run["model/{}".format(self.model_path.split('/')[1])].upload(self.model_path)
                    result_dict={'Validation Loss': best_val_loss,
                                'Stopping Epoch': best_epoch}
                    for i, item in enumerate(best_auroc_valid):
                        result_dict[self.train_ds.pathologies[i]]=item
                    self.run['validation results'] = result_dict
                    with open(self.result_path, "w") as write_file:
                        json.dump(result_dict, write_file, indent=4) 
                else:
                    print('Model has been saved in Epoch %d'%(1+best_epoch) + '. The best score (Loss) obtained is: %.4f'%best_val_loss)

           