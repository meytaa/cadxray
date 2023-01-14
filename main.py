import yaml
from dataloader import read_data
import torch



def main(config):

    train_dataset, valid_dataset, test_dataset = read_data(config['data'], verbose=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], num_workers=config['training']['num_workers'], shuffle=True)
    for i in train_loader:
        i[0]
if __name__ == '__main__':

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)