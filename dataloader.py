from torchvision import transforms as T    
from torchvision import datasets as datasets
import torch
from torch.utils import data
import pandas as pd
import os
from PIL import Image
import numpy as np


class CXR_dataset(data.Dataset):
    def __init__(self, dataset, datapath, csvpath, transforms, mode='train', augmentation_prob=0.9, indices=None):
        
        self.df = pd.read_csv(csvpath,sep=',')
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png", # They are missing
            "216840111366964012989926673512011074122523403_00-163-058.png",
            "216840111366964012959786098432011033083840143_00-176-115.png",
            "216840111366964012558082906712009327122220177_00-102-064.png",
            "216840111366964012339356563862009072111404053_00-043-192.png",
            "216840111366964013076187734852011291090445391_00-196-188.png",
            "216840111366964012373310883942009117084022290_00-064-025.png",
            "216840111366964012283393834152009033102258826_00-059-087.png",
            "216840111366964012373310883942009170084120009_00-097-074.png",
            "216840111366964012819207061112010315104455352_04-024-184.png",
            "216840111366964012819207061112010306085429121_04-020-102.png"]
        self.df = self.df[~self.df["Image Index"].isin(missing)].reset_index()
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.transforms = transforms
        self.datapath = datapath
        self.filenames = self.df["Image Index"].tolist()

        self.image_paths = list(map(lambda x: os.path.join(datapath, x), self.filenames))
        if (indices !=None):
            self.filenames = [self.filenames[i] for i in indices]
            self.image_paths = [self.image_paths[i] for i in indices]
        self.len=len(self.filenames)
        if dataset=='cxr14':
            self.pathologies = list(self.df.columns)[4:]
        elif dataset=='pc':
            self.pathologies = list(self.df.columns)[2:]
        else:
            print('Specify the pathologies for the given dataset!')




    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if np.array(image).max()>255 :
            image = Image.fromarray(np.uint8(np.array(image)/256))
        image = self.transforms(image)
        image_name = self.filenames[index]
        labels = torch.tensor(list(self.df.iloc[index][self.pathologies]))
        return image, labels, image_name
        
    def __len__(self):
        return self.len
    



def get_transforms(im_size):
    transforms = T.Compose([
                T.Resize((im_size, im_size)),
                T.RandomRotation(10),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=(-10, 10), translate=(0., 0.1), scale=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize((0.5), (0.5)),
                ])
    return transforms

def split_dataset(dataset, train_percent=0.65, test_percent=0.25):
    '''
    The function is used for splitting the datasetto train, validation, and test subsets

    Parameters
    ----------
    dataset : torchvision dataset
        the whole dataset


    Returns
    -------
    torchvision datasets
    '''
    dataset_size = len(dataset)
    train_size = int(train_percent*dataset_size)
    test_size = int(test_percent*dataset_size)
    val_size = dataset_size - train_size - test_size
    # Splitting the dataset into subsets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    test_dataset.pathologies = dataset.classes
    train_dataset.pathologies = dataset.classes
    val_dataset.pathologies = dataset.classes
    return train_dataset, val_dataset, test_dataset





def read_data(config, verbose=False):
    '''
    The function handle the data reading for databases and apply transform if needed
    '''
    print(100 * '_')
    print('Reading data!')
    transforms = get_transforms(config['image_size'])
    # Getting addresses from the config parameters
    data_path = config[config['machine']][config['dataset']]['data_path']
    if not config['dataset']=='tb':
        csv_path = config[config['machine']][config['dataset']]['csv_path']
    else:
        csv_path = ''
    
    # tb dataset is accessible via ImageFolder. O.W we use a same functions for datasets.
    if config['dataset'] == 'tb':
        all_dataset = datasets.ImageFolder(data_path, transform=transforms)#, target_transform=target_to_oh
        train_dataset, val_dataset, test_dataset = split_dataset(all_dataset, train_percent=0.65, test_percent=0.25)
        print('dataset created')
    else: # Tested on CXR14 and PC
        train_dataset = CXR_dataset(config['dataset'],
                                datapath=data_path,
                                csvpath= os.path.join(csv_path,'train.csv'),
                                transforms=transforms,
                                mode='train',
                                augmentation_prob= config['aug_prob'])
        test_dataset = CXR_dataset(config['dataset'],
                                datapath=data_path,
                                csvpath= os.path.join(csv_path, 'test.csv'),
                                transforms=transforms,
                                mode='test',
                                augmentation_prob= config['aug_prob'])
        val_dataset = CXR_dataset(config['dataset'],
                                datapath=data_path,
                                csvpath= os.path.join(csv_path, 'dev.csv'),
                                transforms=transforms,
                                mode='dev',
                                augmentation_prob= config['aug_prob'])
    
    print('Dataset name: {} \nDataset size: {}'.format(config['dataset'], len(train_dataset)+len(val_dataset)+len(test_dataset)))
    print()
    if verbose:
        print('Addresses introduced for "{}" running on {} machine. \nImage data path: {} \nCSV file path: {}'.format(config['dataset'], config['machine'], data_path, csv_path))
        print('\n# Train set samples:      ', len(train_dataset), '\n# Test set samples:       ',\
        len(test_dataset), '\n# Validation set samples: ', len(val_dataset) , '\nLoading Completed' +'\n')
    print('Reading Data Completed.')
    print(100 * '_')
    return train_dataset, val_dataset, test_dataset