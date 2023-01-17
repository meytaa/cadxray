import yaml
from dataloader import read_data
import torch
from assemble import compile_nets


def main(config):

    config['train_dataset'], config['valid_dataset'], config['test_dataset'] = read_data(config['data'], verbose=True)
    assembler = compile_nets(config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True)
    for i in train_loader:
        i[0]
    
if __name__ == '__main__':

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)