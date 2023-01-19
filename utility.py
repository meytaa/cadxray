from sklearn.metrics import roc_auc_score
import yaml



def target_to_oh(target, NUM_CLASS):
    one_hot = [0] * NUM_CLASS
    one_hot[target] = 1
    return one_hot


def get_AUROC(label,pred):
    auroc = []
    for i in range(label.shape[1]):
        auroc.append(roc_auc_score(label[:,i],pred[:,i]))
    return auroc


def argparsert2yaml(config):
    config_dict = {}
    config_dict['neptune'] = {}
    config_dict['neptune']['API-KEY'] = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGRmYzUzNS0yZGY3LTRmNWEtYWYwNi03YzM2MWU0MmM5YzMifQ=='

    config_dict['model'] = {}
    config_dict['model']['model_name'] = config.model_type
    config_dict['model']['init_weights'] = config.init_weights
    config_dict['model']['output_ch'] = config.output_ch

    config_dict['training'] = {}
    config_dict['training']['num_epochs'] = config.num_epochs
    config_dict['training']['batch_size'] = config.batch_size
    config_dict['training']['num_workers'] = config.num_workers
    config_dict['training']['lr'] = config.lr
    config_dict['training']['lr_decay'] = config.lr_decay
    config_dict['training']['tr_strategy'] = config.tr_strategy
    config_dict['training']['loss_fcn'] = config.loss_fcn
    config_dict['training']['mode'] = config.mode
    config_dict['training']['generalization'] = config.generalization
    config_dict['training']['load_model_id'] = config.load_model_id
    config_dict['training']['validation_flag'] = config.validation_flag

    config_dict['log'] = {}
    config_dict['log']['model_path'] = config.model_path
    config_dict['log']['result_path'] = config.result_path
    config_dict['log']['layer_path'] = config.layer_path

    config_dict['data'] = {}
    config_dict['data']['machine'] = config.machine
    config_dict['data']['train_dataset'] = config.train_dataset
    config_dict['data']['test_dataset'] = config.test_dataset
    config_dict['data']['image_size'] = config.image_size
    config_dict['data']['aug_prob'] = config.aug_prob

    config_dict['data']['local'] = {}
    config_dict['data']['local']['tb'] = {}
    config_dict['data']['local']['cxr14'] = {}
    config_dict['data']['local']['pc'] = {}

    config_dict['data']['cc'] = {}
    config_dict['data']['cc']['tb'] = {}
    config_dict['data']['cc']['cxr14'] = {}
    config_dict['data']['cc']['pc'] = {}

    config_dict['data']['local']['tb']['data_path'] = '/media/mohammad/Windows/Users/Mohammad/Desktop/Files/Thesis/Pytorch/data/TB_Chest_Radiography_Database'

    config_dict['data']['local']['cxr14']['data_path'] = '/media/mohammad/Windows/Users/Mohammad/Desktop/Files/Thesis/Pytorch/data/ChestXray14/images'
    config_dict['data']['local']['cxr14']['csv_path'] = '/media/mohammad/Windows/Users/Mohammad/Desktop/Files/Thesis/Pytorch/data/ChestXray14/new_split_xray14'

    config_dict['data']['local']['pc']['data_path'] = '/media/mohammad/Windows/Users/Mohammad/Desktop/Files/Thesis/Pytorch/data/Padchest/PC/images-224'
    config_dict['data']['local']['pc']['csv_path'] = '/media/mohammad/Windows/Users/Mohammad/Desktop/Files/Thesis/Pytorch/data/Padchest/PC/split_PC_'


    config_dict['data']['cc']['tb']['data_path'] = '/home/meyta/scratch/Datasets/TB_Chest_Radiography_Database'

    config_dict['data']['cc']['cxr14']['data_path'] = '/home/meyta/scratch/Datasets/ChestXray14/images'
    config_dict['data']['cc']['cxr14']['csv_path'] = '/home/meyta/scratch/Datasets/ChestXray14/new_split_xray14'

    config_dict['data']['cc']['pc']['data_path'] = '/home/meyta/scratch/Datasets/PC/images-224'
    config_dict['data']['cc']['pc']['csv_path'] = '/home/meyta/scratch/Datasets/PC/split_PC'

    with open("config.yaml", "w") as f:
        yaml.dump(config_dict, f)
    
    return config_dict
    




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.elements = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.elements.append(val)


class Appender(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.elements = []
    def update(self, val):
        self.elements.append(val)