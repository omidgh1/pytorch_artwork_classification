from torch.utils.data import DataLoader


class CustomDataLoader:
    """
    CustomDataLoader is a utility class designed to streamline the usage of PyTorch's DataLoader for
    handling datasets in deep learning projects. It encapsulates a single DataLoader
    object and provides methods for convenient iteration over the dataset as well as querying its length.
    This class simplifies the process of data loading and batching,
    making it easier to integrate datasets into PyTorch models and training loops.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)



"""
from CustomDataset import CustomDataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class CustomDataLoader:
    def __init__(self, paths: list, labels: list):
        self.paths = paths
        self.labels = labels

    def construction(self, batch_size) -> DataLoader:
        datasets: list[CustomDataset] = [CustomDataset(path=x, labels=self.labels, transform=None) for x in self.paths]
        dataset: Dataset = ConcatDataset(datasets)
        dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def __len__(self):
        batch_size: int = 4
        dataloader: DataLoader = self.construction(batch_size=batch_size)
        length: int = dataloader.__len__()
        return length
"""