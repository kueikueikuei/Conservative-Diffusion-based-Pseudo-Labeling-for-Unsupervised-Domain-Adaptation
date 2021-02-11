from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
class Office31(Dataset):
    def __init__(self, root, datasetname):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        
        
        self.transform = transforms.Compose(
            [transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])

        # load image path and annotations
        df_train = pd.read_csv(root+datasetname+"train.csv")
        df_test = pd.read_csv(root+datasetname+"test.csv")
        self.img = torch.load(root+datasetname+'.pt')
        df = pd.concat([df_train,df_test])
#         self.trainx = df["0"].to_numpy()
        self.trainy = df["1"].to_numpy()
        self.conf = np.zeros(len(self.trainy))
        self.ps = np.zeros(len(self.trainy))
        assert self.img.shape[0] == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
#         imgpath = self.trainx[index]
#         img = Image.open("../"+imgpath).convert('RGB')
        img = self.img[index]
        lbl = int(self.trainy[index])
#         img = self.transform(img)
        conf = self.conf[index]
        ps = self.ps[index]
#         ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainy)

class Office31Dataset(TorchvisionDataset):

    def __init__(self, root_path: str , datasetname: str):
        super().__init__()
        
        self.n_classes = 31  # 0: normal, 1: outlier

        self.train_set = Office31(root_path, datasetname)
        
    def __len__(self):

        return len(self.train_set)