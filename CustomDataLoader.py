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
