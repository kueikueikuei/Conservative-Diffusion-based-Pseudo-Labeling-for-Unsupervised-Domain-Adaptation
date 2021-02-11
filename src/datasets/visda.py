from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
class VISDA17(Dataset):
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
        dataset={"visda_src":"train/","visda_tar":"validation/","visda_test":"test/"}
        df_train = pd.read_csv(root+dataset[datasetname]+"image_list.txt",sep='\s+',header=None)
        print(df_train.groupby([1]).head(1))
        self.path = root+dataset[datasetname]
        self.img = df_train[0].to_numpy()
        self.trainy = df_train[1].to_numpy()
#         radomidx = np.random.permutation(len(self.trainy))[:int(len(self.trainy)/4)]
        
#         self.img=self.img[radomidx]
#         self.trainy=self.trainy[radomidx]
# #         self.path=self.path[radomidx]
        self.conf = np.zeros(len(self.trainy))
        self.ps = np.zeros(len(self.trainy))
        assert len(self.img) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.img[index]
        img = Image.open(self.path+imgpath).convert('RGB')
        lbl = int(self.trainy[index])
        img = self.transform(img)
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.img)

class VISDA17Dataset(TorchvisionDataset):

    def __init__(self, root_path: str , datasetname: str):
        super().__init__()
        
        self.n_classes = 12  

        self.train_set = VISDA17(root_path, datasetname)
        
    def __len__(self):

        return len(self.train_set)