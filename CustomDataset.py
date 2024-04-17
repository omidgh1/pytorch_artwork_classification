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
        label: np.array = np.array([float(x) for x in self.path[13:]]).astype('float64')
        return image, label

    def __len__(self):
        return len(self.images)
