import torch
import torch.nn as nn
class CNN_Model(nn.Module):
    """
    CNN_Model is a custom convolutional neural network (CNN) model designed for image classification tasks.
    It consists of two convolutional layers followed by two fully connected layers.
    The model expects input images with three color channels (RGB) and outputs logits
    for classification into five classes. Each convolutional layer is activated
    by Rectified Linear Unit (ReLU) activation functions and followed by max-pooling layers for downsampling.
    The fully connected layers perform classification based on the features extracted by the convolutional layers.
    """
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 5)  # Assuming 5 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 56 * 56)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


"""
from torchvision.models import AlexNet
from torch.nn import *
import torch.nn.functional as f


class Model(AlexNet):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.layer3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = Linear(in_features=57600, out_features=512)
        self.fc2 = Linear(in_features=512, out_features=10)  # Assuming 10 output cla

    def forward(self, x):
        x = self.pool(f.relu(self.layer1(x)))
        x = self.pool(f.relu(self.layer2(x)))
        x = self.pool(f.relu(self.layer3(x)))
        x = x.view(-1, 57600)     # Reshape for fully connected layer
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""