import yaml, argparse
from dataloader import read_data
import torch
from assemble import compile_nets
import neptune.new as neptune
from utility import argparsert2yaml
#github token: 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

def main(config):
    config['train_dataset'], config['valid_dataset'], config['test_dataset'] = read_data(config['data'], verbose=True)
    assembler = compile_nets(config)
    assembler.train()
    train_loader = torch.utils.data.DataLoader(config['train_dataset'], batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True)
    # for i in train_loader:
    #     print(i)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=92)

    # training hyper-parameters
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=32) 

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6) 
    parser.add_argument('--lr_decay', type=float, default=5e-7) 
    parser.add_argument('--aug_prob', type=float, default=1.)
    parser.add_argument('--tr_strategy', type=str, default='off',help= 'off/tune')
    parser.add_argument('--generalization', type=str, default='base',help= 'base')
    parser.add_argument('--load_model_id', type=str, default='230116-289957',help= '')
    parser.add_argument('--validation_flag', type=str, default=True)
    
    parser.add_argument('--init_weights', type=str, default='scratch',help= 'Imagenet/normal/xavier/kaiming/orthogonal')
    parser.add_argument('--loss_fcn', type=str, default='bce',help= 'bce')

    
    # misc
    parser.add_argument('--machine', type=str, default='local', help='local/cc')
    parser.add_argument('--mode', type=str, default='train', help='train/dev/test')
    parser.add_argument('--model_type', type=str, default='densenet121', help='Densenet/Efficientnet/Densenet-chex')
    parser.add_argument('--model_path', type=str, default='models')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--layer_path', type=str, default='layers')
    parser.add_argument('--train_dataset', type=str, default='pc', help='cxr14/pc/tb')
    parser.add_argument('--test_dataset', type=str, default='pc', help='cxr14/pc/tb')
    config = parser.parse_args()
    config_dict = argparsert2yaml(config)
    # with open('config.yml', 'r') as file:
    #     config = yaml.safe_load(file)
    main(config_dict)