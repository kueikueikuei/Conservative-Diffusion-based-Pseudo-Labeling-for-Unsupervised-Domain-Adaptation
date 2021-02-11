from .digit import DigitDataset
from .office31 import Office31Dataset
from .imageclef import ImageCLEFDataset
from .visda import VISDA17Dataset
from base.torchvision_dataset import TorchvisionDataset
# from .cifar import CIFAR_Dataset
# from .stl import STL_Dataset
import torch
import numpy as np
def load_dataset(dataset_name):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'svhn', 'mnistm', 'syn', 'usps', 'amazon', 'dslr', 'webcam','digit_other','b','c','i','p','other','visda_src','visda_tar','visda_test')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist' or dataset_name == 'svhn' or dataset_name == 'mnistm' or dataset_name == 'syn' or dataset_name == 'usps' or dataset_name == 'digit_other':
        dataset = DigitDataset("../datafolder/DIGIT/", dataset_name)
        dataset.domain="digit"
    if dataset_name == 'amazon' or dataset_name == 'dslr' or dataset_name == 'webcam' or dataset_name == 'other':
        dataset = Office31Dataset("../datafolder/OFFICE31/", dataset_name) 
        dataset.domain="office31"
        
    if dataset_name == 'b' or dataset_name == 'c' or dataset_name == 'i'or dataset_name == 'p':
        dataset = ImageCLEFDataset("../datafolder/image-clef/", dataset_name) 
        dataset.domain="image-clef"
    if dataset_name == "visda_src" or dataset_name == "visda_tar" or dataset_name == "visda_test":
        dataset = VISDA17Dataset("../datafolder/VISDA/", dataset_name) 
        dataset.domain="visda"
    return dataset

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        
        conf = np.empty((0))
        ps = np.empty((0))
        n,c,h,w = datasets[0].trainx.shape
        trainx = torch.empty((0, c,h,w))
        trainy = np.empty((0))
        for dataset in datasets:
            conf = np.concatenate((conf,dataset.conf),axis=0)
            ps = np.concatenate((ps,dataset.ps),axis=0)
            trainy = np.concatenate((trainy,dataset.trainy),axis=0)
            trainx = torch.cat((trainx, dataset.trainx), 0)
            
        self.conf = conf
        self.ps = ps
        self.trainx = trainx
        self.trainy = trainy
#         self.datasets = datasets
    def __getitem__(self, index):
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps

    def __len__(self):
        n,c,h,w = self.trainx.shape
        return n
    
class DigitConDataset(TorchvisionDataset):

    def __init__(self, data_path: str , dataset_name_list: list):
        super().__init__()
        self.root_path = data_path
        self.n_classes = 10
        all_datasets=[]
        for dataset_name in dataset_name_list:
            all_datasets.append(load_dataset(dataset_name, data_path).train_set)
        self.train_set = ConcatDataset(all_datasets)
        