from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    """
    CustomDataset is designed to facilitate the loading of image data for classification tasks.
    It utilizes PyTorch's built-in functionalities for dataset handling,
    specifically leveraging the ImageFolder dataset class from torchvision.
    This class automatically loads images and their corresponding labels
    from a directory structure where each subdirectory represents a different class.
    Additionally, CustomDataset supports data transformation using PyTorch's transformation pipeline,
    enabling efficient preprocessing of images, including resizing, normalization, and augmentation.
    """
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.labels = [label for _, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.labels[idx]


"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path: str, labels, transform=None):
        self.transform = transform
        self.labels = labels
        self.path: str = path
        self.images: list = os.listdir(path)

    def __getitem__(self, idx: int) -> tuple[np.array, np.array]:
        image_path: str = self.path + '/' + self.images[idx]
        print('Immagine:', self.images[idx])
        if cv2.imread(filename=image_path, flags=0) is not None:
            image_path: str = self.path + '/' + self.images[idx]
        else:
            image_path: str = self.path + '/' + self.images[idx - 1]
        img: np.array = cv2.imread (filename=image_path, flags=0).astype(np.float64)
        image_resize: np.array = cv2.resize(img, dsize=(240, 240))
        image: np.array = (image_resize - image_resize.min()) / (image_resize.max() - image_resize.min())
        label: np.array = np.array([float(x) for x in self.path[15:]]).astype('float64') #13
        return image, label

    def __len__(self):
        return len(self.images)
"""