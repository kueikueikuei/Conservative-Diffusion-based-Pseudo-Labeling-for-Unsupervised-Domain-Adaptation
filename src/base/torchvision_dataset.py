from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
import torch

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self):
        super().__init__()

    def loaders(self, batch_size: int, shuffle=True, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)
        return train_loader
