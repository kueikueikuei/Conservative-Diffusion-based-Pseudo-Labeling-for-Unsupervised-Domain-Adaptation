from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self):
        super().__init__()
        self.root = None  # root path to data

        self.n_classes = None

        self.train_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0, pin_memory=True) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__
