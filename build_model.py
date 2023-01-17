import torchvision
from torchvision.models import densenet121, efficientnet_b7, densenet161, efficientnet_b2, mobilenet_v3_large#, vit_b_32, resnet18
import torch.nn as nn
import numpy as np
import torch
import os, re
from torch.nn import init

class build_model(object):
    def __init__(self, config) -> None:
        self.model_name = config['model_name']
        self.init_weights = config['init_weights']
        self.output_size = config['output_ch']
        self.create_model()
        self.config_classifier()
        self.initiate_model()
        
    def create_model(self):
        self.model = eval('{}(pretrained={})'.format(self.model_name, self.init_weights=='imagenet'))
        if self.init_weights=='imagenet':
            print('Weights trained on {} is loaded!'.format(self.init_weights))

    def initiate_model(self, gain=0.02):
        if not (self.init_weights=='pretrained' or self.init_weights=='imagenet'):
            def init_func(m):
                classname = m.__class__.__name__
                if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if self.init_weights == 'normal':
                        init.normal_(m.weight.data, 0.0, gain)
                    elif self.init_weights == 'xavier':
                        init.xavier_normal_(m.weight.data, gain=gain)
                    elif self.init_weights == 'kaiming':
                        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    elif self.init_weights == 'orthogonal':
                        init.orthogonal_(m.weight.data, gain=gain)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % self.init_weights)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)
                elif classname.find('BatchNorm2d') != -1:
                    init.normal_(m.weight.data, 1.0, gain)
                    init.constant_(m.bias.data, 0.0)
            self.model.apply(init_func)
            print("Model's weights are Initialized using {} method.".format(self.init_weights))


    def get_modelname(self, model_id):
        model_list = os.listdir('models/')
        r = re.compile("^.+{}.pkl$".format(model_id))#author.handle:.*/*.csv")# Working with one type of table with the same format
        model_name = list(filter(r.match, model_list))[0]
        return "models/{}".format(model_name)

    def config_classifier(self):
        if self.model_name == 'densenet121':
            self.model.classifier = nn.Linear(1024,self.output_size)
        elif self.model_name == 'densenet161':
            self.model.classifier = nn.Linear(2208,self.output_size)
        elif self.model_name =='efficientnet_b7':
            self.model.classifier[1] = nn.Linear(2560, self.output_size)
        elif self.model_name =='efficientnet_b2':
            self.model.classifier[1] = nn.Linear(1408, self.output_size)
        elif self.model_name =='mobilenet_v3_large':
            self.model.classifier[-1] = nn.Linear(1280, self.output_size)

    def load_pretrain(self, model_id):
        if self.init_weights=='pretrained':
            model_name = self.get_modelname(model_id)
            self.model.load_state_dict(torch.load(model_name))
        return model_name

    def display(self):
        """Print out the network information."""
        num_trainable_params = 0
        num_params = 0
        for p in self.model.parameters():
            if p.requires_grad==True:
                num_params += p.numel()
                num_trainable_params += p.numel()
            if p.requires_grad==False:
                num_params += p.numel()
                pass
        name = self.model.__class__.__name__
        model_type = self.model_name
        print('\nModel Name:{}({})'.format(name, model_type))
        print("Training parameters: {}\nAll parameters:      {}".format(num_trainable_params, num_params))



if __name__ == '__main__':

    config={}
    config['model_name']='densenet121'
    config['init_weights'] = 'pretrained'
    config['output_ch']=2
    build_model(config)