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
        self.lr = float(config['training']['lr'])
        self.lr_decay = float(config['training']['lr_decay'])

        
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
        model_creator = build_model(config['model'])
        if config['training']['load_model_id'] and config['model']['init_weights']=='pretrained':
            try:
                pretrained_model_name = model_creator.load_pretrain(model_id = config['training']['load_model_id'])
                print('Pretraied model loaded\nModel ID: {}\nModel path: {}'.format(config['training']['load_model_id'], pretrained_model_name))
            except:
                print('Pretrained model with ID={} does not exist or not matched to the model!.'.format(config['training']['load_model_id']))
        self.model = model_creator.model
        model_creator.display()
        self.set_optimizer()
        print('so far done!')




    def set_criterion(self):
        if self.loss_fcn=='bce':
            self.criterion_class = nn.BCELoss()
        print('Loss function is set to {}.'.format(self.criterion_class))

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)



    def train(self):
        writer = SummaryWriter(os.path.join('runs','runs_'+self.model_type,self.store_name))
        

        best_val_loss = 500
        train_epoch_losses = []
        valid_epoch_losses = []
        train_itr = int(len(self.train_loader.dataset)/self.batch_size)
        valid_itr = int(len(self.valid_loader.dataset)/self.batch_size)
        tr_itr_c= 0
        val_itr_c= 0
        middle_layer_num = int(0.5*len(self.model.module.features)) if torch.cuda.device_count() > 1 else int(0.5*len(self.model.features))
        l_names = ['early', 'middle', 'late']
        l_nums = [2, middle_layer_num, -1]
        for epoch in range(self.num_epochs):
            self.model.train()
            d_label = []
            d_pred = []
            epoch_loss = 0
            early_layer = []
            middle_layer = []
            late_layer = []
            print('Training')
            start_time = time.time()
            for i, (image, label) in enumerate (self.train_loader):
                label = torch.nn.functional.one_hot(label, num_classes=2)
                image = image.to(self.device)
                label = label.float().to(self.device)
                pred = self.model(image)
                early_layer = self.model.module.features[:2](image).detach().cpu().numpy().astype('float16')\
                     if torch.cuda.device_count() > 1 else self.model.features[:2](image).detach().cpu().numpy().astype('float16')
                # torch.save(early_layer, os.path.join(self.layer_dir, 'EP_{epoch}-itr_{i}-L_2.pt'.format(epoch=epoch, i=i)))
                np.save(os.path.join(self.layer_dir, 'EP_{epoch}-itr_{i}-L_2'.format(epoch=epoch, i=i)), early_layer)
                middle_layer = self.model.module.features[:-middle_layer_num](image).detach().cpu().numpy().astype('float16') \
                     if torch.cuda.device_count() > 1 else self.model.features[:-middle_layer_num](image).detach().cpu().numpy().astype('float16')
                # torch.save(middle_layer, os.path.join(self.layer_dir, 'EP_{epoch}-itr_{i}-L_{middle_layer_num}.pt'.format(epoch=epoch, i=i, middle_layer_num=middle_layer_num)))
                np.save(os.path.join(self.layer_dir, 'EP_{epoch}-itr_{i}-L_{middle_layer_num}.npy'.format(epoch=epoch, i=i, middle_layer_num=middle_layer_num)), middle_layer)
                late_layer = self.model.module.features(image).detach().cpu().numpy().astype('float16')\
                     if torch.cuda.device_count() > 1 else self.model.features(image).detach().cpu().numpy().astype('float16')
                np.save(os.path.join(self.layer_dir, 'EP_{epoch}-itr_{i}-L_-1.npy'.format(epoch=epoch, i=i)), late_layer)

                pred = torch.sigmoid(pred)
                loss = self.criterion_class(pred,label)
                d_label.append(label.detach().cpu().numpy())
                d_pred.append(pred.detach().cpu().numpy())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()*image.size(0)
                end_time = time.time()
                print('Epoch [ %4d/%d ] - Iteration [ %5d / %d ] --> Loss = %.4f - iteration time:%.4f '%(epoch+1, self.num_epochs, i,train_itr,loss.item(), end_time-start_time))
                start_time = end_time
                tr_itr_c +=1
                writer.add_scalar('Iteration Loss/Train', loss.item(), tr_itr_c)

            # itr_files = os.listdir(self.layer_dir)
            # for layer_num in l_nums:
            #     r = re.compile(".*L_{}.*".format(layer_num))
            #     associate_list = list(filter(r.match, itr_files))
                # list_of_tensors = [torch.load(os.path.join(self.layer_dir,file)) for file in associate_list]
                # associate_tensor = torch.concat(list_of_tensors, 0)
                # del list_of_tensors
                # torch.save(associate_tensor, os.path.join(self.epoch_layer_dir, "EP_{}-L_{}.pt".format(epoch, layer_num)))
   


            epoch_loss /=len(self.train_loader.dataset)
            d_pred = np.concatenate(d_pred)
            d_label = np.concatenate(d_label)
            train_epoch_losses.append(epoch_loss)
            auroc_train = get_AUROC(d_label,d_pred)
            for i in range(len(auroc_train)):
                writer.add_scalar('AUCROC '+ self.train_ds.pathologies[i]+ '/Train', auroc_train[i], epoch)
            writer.add_scalar('Loss/Epoch(Train)', epoch_loss, epoch)
            
            self.model.eval()
            d_val_label = []
            d_val_pred = []
            epoch_val_loss = 0
            print('Validation')