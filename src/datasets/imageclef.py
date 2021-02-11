from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
class ImageCLEF(Dataset):
    def __init__(self, root, datasetname):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        
        
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])

        # load image path and annotations
        df_train = pd.read_csv(root+"list/"+datasetname+"List.txt",sep='\s+',header=None)
#         self.path = root
#         self.trainx = df_train[0].to_numpy()
        self.img = torch.load(root+datasetname+'.pt')
        self.trainy = df_train[1].to_numpy()
        self.conf = np.zeros(len(self.trainy))
        self.ps = np.zeros(len(self.trainy))
        assert self.img.shape[0] == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.img[index]
        lbl = int(self.trainy[index])
#         img = self.transform(img)
        conf = self.conf[index]
        ps = self.ps[index]
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainy)

class ImageCLEFDataset(TorchvisionDataset):

    def __init__(self, root_path: str , datasetname: str):
        super().__init__()
        
        self.n_classes = 12  

        self.train_set = ImageCLEF(root_path, datasetname)
        
    def __len__(self):

        return len(self.train_set)