from base.torchvision_dataset import TorchvisionDataset
from torchvision import datasets, transforms
from scipy.io import loadmat
from .preprocessing import s2t, u2t, resize_to_32
import pickle as pkl
import torch
import numpy as np
import os
import h5py
import torch.utils.data as data

from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torch
from PIL import Image
import pandas as pd
import numpy as np
class mnist(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        train = loadmat(os.path.join(self.root_path, 'mnist32_train.mat'))
        test = loadmat(os.path.join(self.root_path, 'mnist32_test.mat'))
        
        trainx, trainy = train['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), train['y'].reshape(-1)
        testx, testy = test['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), test['y'].reshape(-1)
        
        trainx = s2t(trainx)
        testx = s2t(testx)
        trainx = np.concatenate((trainx,testx),axis=0)
        trainy = np.concatenate((trainy,testy),axis=0)
        self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)
class svhn(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        train = loadmat(os.path.join(self.root_path, 'train_32x32.mat'))
        test = loadmat(os.path.join(self.root_path, 'test_32x32.mat'))

        trainx, trainy = train['X'].transpose((3, 2, 0, 1)), train['y'].reshape(-1)%10
        testx, testy = test['X'].transpose((3, 2, 0, 1)), test['y'].reshape(-1)%10
        trainx = u2t(trainx)
        testx = u2t(testx)
        trainx = np.concatenate((trainx,testx),axis=0)
        trainy = np.concatenate((trainy,testy),axis=0)
        self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)
    
class mnistm(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        data = pkl.load(open(os.path.join(self.root_path, 'mnistm_data.pkl'), "rb"))
        labels = pkl.load(open(os.path.join(self.root_path, 'mnistm_labels.pkl'), "rb"))

        trainx, trainy = data['train'], labels['train']
        validx, validy = data['valid'], labels['valid']
        testx, testy = data['test'], labels['test']
        trainx = np.concatenate((trainx, validx), axis=0)
        trainy = np.concatenate((trainy, validy), axis=0)
        trainx = resize_to_32(trainx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
        testx = resize_to_32(testx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
        trainx = u2t(trainx)
        testx = u2t(testx)
        trainx = np.concatenate((trainx,testx),axis=0)
        trainy = np.concatenate((trainy,testy),axis=0)
        self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)
class syn(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        train = loadmat(os.path.join(self.root_path, 'synth_train_32x32.mat'))
        test = loadmat(os.path.join(self.root_path, 'synth_test_32x32.mat'))
        print(train['X'].shape)
        trainx, trainy = train['X'].transpose((3, 2, 0, 1)), train['y'].reshape(-1)%10
        testx, testy = test['X'].transpose((3, 2, 0, 1)), test['y'].reshape(-1)%10
        trainx = u2t(trainx)
        testx = u2t(testx)
        trainx = np.concatenate((trainx,testx),axis=0)
        trainy = np.concatenate((trainy,testy),axis=0)
        self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)
class usps(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        with h5py.File("../data/DIGIT/usps.h5", 'r') as hf:
            train = hf.get('train')
            trainx = train.get('data')[:]

            trainx = np.tile(trainx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
        #     trainx = resize_to_32(trainx).transpose((0,3,1,2))
            trainy = train.get('target')[:]
            test = hf.get('test')
            testx = test.get('data')[:]

            testy = test.get('target')[:]
            testx = np.tile(testx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
            
            trainx = resize_to_32(trainx).transpose((0,3,1,2))
            testx = resize_to_32(testx).transpose((0,3,1,2))
            trainx = u2t(trainx)
            testx = u2t(testx)
            trainx = np.concatenate((trainx,testx),axis=0)
            trainy = np.concatenate((trainy,testy),axis=0)
            self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)

    
    
class other(Dataset):
    def __init__(self, root):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
     
        self.root_path = root
        
        train = loadmat(os.path.join(self.root_path, 'mnist32_train.mat'))
        test = loadmat(os.path.join(self.root_path, 'mnist32_test.mat'))
        
        trainx, trainy = train['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), train['y'].reshape(-1)
        testx, testy = test['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), test['y'].reshape(-1)
        
        trainx = s2t(trainx)
        testx = s2t(testx)
        trainx_m = np.concatenate((trainx,testx),axis=0)
        trainy_m = np.concatenate((trainy,testy),axis=0)
        
        
        data = pkl.load(open(os.path.join(self.root_path, 'mnistm_data.pkl'), "rb"))
        labels = pkl.load(open(os.path.join(self.root_path, 'mnistm_labels.pkl'), "rb"))

        trainx, trainy = data['train'], labels['train']
        validx, validy = data['valid'], labels['valid']
        testx, testy = data['test'], labels['test']
        trainx = np.concatenate((trainx, validx), axis=0)
        trainy = np.concatenate((trainy, validy), axis=0)
        trainx = resize_to_32(trainx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
        testx = resize_to_32(testx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
        trainx = u2t(trainx)
        testx = u2t(testx)
        trainx_mm = np.concatenate((trainx,testx),axis=0)
        trainy_mm = np.concatenate((trainy,testy),axis=0)
        
        with h5py.File("../data/DIGIT/usps.h5", 'r') as hf:
            train = hf.get('train')
            trainx = train.get('data')[:]

            trainx = np.tile(trainx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
        #     trainx = resize_to_32(trainx).transpose((0,3,1,2))
            trainy = train.get('target')[:]
            test = hf.get('test')
            testx = test.get('data')[:]

            testy = test.get('target')[:]
            testx = np.tile(testx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
            
            trainx = resize_to_32(trainx).transpose((0,3,1,2))
            testx = resize_to_32(testx).transpose((0,3,1,2))
            trainx = u2t(trainx)
            testx = u2t(testx)
            trainx_u = np.concatenate((trainx,testx),axis=0)
            trainy_u = np.concatenate((trainy,testy),axis=0)
            
        train = loadmat(os.path.join(self.root_path, 'synth_train_32x32.mat'))
        test = loadmat(os.path.join(self.root_path, 'synth_test_32x32.mat'))

        trainx, trainy = train['X'].transpose((3, 2, 0, 1)), train['y'].reshape(-1)%10
        testx, testy = test['X'].transpose((3, 2, 0, 1)), test['y'].reshape(-1)%10
        trainx = u2t(trainx)
        testx = u2t(testx)
        trainx_s = np.concatenate((trainx,testx),axis=0)
        trainy_s = np.concatenate((trainy,testy),axis=0)
        trainx = np.concatenate((trainx_m,trainx_mm),axis=0)
        trainx = np.concatenate((trainx,trainx_u),axis=0)
        trainy = np.concatenate((trainy_m,trainy_mm),axis=0)
        trainy = np.concatenate((trainy,trainy_u),axis=0)
        trainx = np.concatenate((trainx,trainx_s),axis=0)
        trainy = np.concatenate((trainy,trainy_s),axis=0)
        self.trainx = torch.Tensor(trainx)
        self.trainy = trainy
        self.conf = np.zeros(len(self.trainx))
        self.ps = np.zeros(len(self.trainx))
        assert len(self.trainx) == len(self.trainy), 'mismatched length!'

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = self.trainx[index]
        lbl = int(self.trainy[index])
        conf = self.conf[index]
        ps = int(self.ps[index])
        return img, lbl, conf,index,ps


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.trainx)
    

        
        
class DigitDataset(TorchvisionDataset):

    def __init__(self, root: str , dataset_name: str):
        super().__init__()
        self.root_path = root
        self.n_classes = 10
        if dataset_name == 'mnist':
            self.train_set = mnist(root)
        elif dataset_name == 'svhn':
            self.train_set = svhn(root)
        elif dataset_name == 'mnistm':
            self.train_set = mnistm(root)
        elif dataset_name == 'syn':
            self.train_set = syn(root)
        elif dataset_name == 'usps':
            self.train_set = usps(root)
        elif dataset_name == 'digit_other':
            self.train_set = other(root)
        def __len__(self):
        
            return len(self.train_set)
#     def load_mnist(self):
#         train = loadmat(os.path.join(self.root_path, 'mnist32_train.mat'))
#         test = loadmat(os.path.join(self.root_path, 'mnist32_test.mat'))
        
#         trainx, trainy = train['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), train['y'].reshape(-1)
#         testx, testy = test['X'].reshape(-1, 32, 32, 3).astype('float32').transpose((0, 3, 1, 2)), test['y'].reshape(-1)
        
#         trainx = s2t(trainx)
#         testx = s2t(testx)
#         trainx = np.concatenate((trainx,testx),axis=0)
#         trainy = np.concatenate((trainy,testy),axis=0)
#         trainx = torch.Tensor(trainx)
#         trainy = torch.LongTensor(trainy)
#         trainx = torch.utils.data.TensorDataset(trainx, trainy.view(-1))
#         return trainx

        
#     def load_svhn(self):
        
#         train = loadmat(os.path.join(self.root_path, 'train_32x32.mat'))
#         test = loadmat(os.path.join(self.root_path, 'test_32x32.mat'))

#         trainx, trainy = train['X'].transpose((3, 2, 0, 1)), train['y'].reshape(-1)%10
#         testx, testy = test['X'].transpose((3, 2, 0, 1)), test['y'].reshape(-1)%10
#         trainx = u2t(trainx)
#         testx = u2t(testx)
#         trainx = np.concatenate((trainx,testx),axis=0)
#         trainy = np.concatenate((trainy,testy),axis=0)
#         trainx = torch.Tensor(trainx)
#         trainy = torch.LongTensor(trainy)
#         trainx = torch.utils.data.TensorDataset(trainx, trainy.view(-1))
#         return trainx

    
#     def load_mnistm(self):
        
#         data = pkl.load(open(os.path.join(self.root_path, 'mnistm_data.pkl'), "rb"))
#         labels = pkl.load(open(os.path.join(self.root_path, 'mnistm_labels.pkl'), "rb"))

#         trainx, trainy = data['train'], labels['train']
#         validx, validy = data['valid'], labels['valid']
#         testx, testy = data['test'], labels['test']
#         trainx = np.concatenate((trainx, validx), axis=0)
#         trainy = np.concatenate((trainy, validy), axis=0)
#         trainx = resize_to_32(trainx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
#         testx = resize_to_32(testx.reshape(-1, 28, 28, 3)).transpose((0,3,1,2))
#         trainx = u2t(trainx)
#         testx = u2t(testx)
#         trainx = np.concatenate((trainx,testx),axis=0)
#         trainy = np.concatenate((trainy,testy),axis=0)
#         trainx = torch.Tensor(trainx)
#         trainy = torch.LongTensor(trainy)
#         trainx = torch.utils.data.TensorDataset(trainx, trainy.view(-1))
#         return trainx

    
#     def load_syn(self):
        
#         train = loadmat(os.path.join(self.root_path, 'train_32x32.mat'))
#         test = loadmat(os.path.join(self.root_path, 'test_32x32.mat'))

#         trainx, trainy = train['X'].transpose((3, 2, 0, 1)), train['y'].reshape(-1)
#         testx, testy = test['X'].transpose((3, 2, 0, 1)), test['y'].reshape(-1)

#         trainx = u2t(trainx)
#         testx = u2t(testx)
#         trainx = np.concatenate((trainx,testx),axis=0)
#         trainy = np.concatenate((trainy,testy),axis=0)
#         trainx = torch.Tensor(trainx)
#         trainy = torch.LongTensor(trainy)
#         trainx = torch.utils.data.TensorDataset(trainx, trainy.view(-1))
#         return trainx

#     def load_usps(self):
#         with h5py.File("../data/DIGIT/usps.h5", 'r') as hf:
#             train = hf.get('train')
#             trainx = train.get('data')[:]

#             trainx = np.tile(trainx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
#         #     trainx = resize_to_32(trainx).transpose((0,3,1,2))
#             trainy = train.get('target')[:]
#             test = hf.get('test')
#             testx = test.get('data')[:]

#             testy = test.get('target')[:]
#             testx = np.tile(testx.reshape(-1, 16, 16, 1).astype('float32'),(1,1,1,3))
            
#             trainx = resize_to_32(trainx).transpose((0,3,1,2))
#             testx = resize_to_32(testx).transpose((0,3,1,2))
#             trainx = u2t(trainx)
#             testx = u2t(testx)
#             trainx = np.concatenate((trainx,testx),axis=0)
#             trainy = np.concatenate((trainy,testy),axis=0)
#             trainx = torch.Tensor(trainx)
#             trainy = torch.LongTensor(trainy)
#             trainx = torch.utils.data.TensorDataset(trainx, trainy.view(-1))
#         return trainx
