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
