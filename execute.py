import torchvision.transforms as transforms  # Import transformations for image preprocessing
from CustomDataLoader import CustomDataLoader  # Import custom data loader class
from CustomDataset import CustomDataset  # Import custom dataset class
from Model import CNN_Model  # Import custom CNN model class
import torch.nn as nn  # Import neural network modules from PyTorch
import torch.optim as optim  # Import optimization modules from PyTorch
from Training import Trainer  # Import custom trainer class for model training

# Data Preparation
train_dir = 'training_set'  # Define path to the training dataset directory
val_dir = 'validation_set'  # Define path to the validation dataset directory

# Data Transformation
transform = transforms.Compose([  # Compose a sequence of image transformations
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image pixel values
])

# Create datasets and data loaders
train_dataset = CustomDataset(train_dir, transform=transform)  # Create training dataset with defined transformation
val_dataset = CustomDataset(val_dir, transform=transform)  # Create validation dataset with defined transformation

train_loader = CustomDataLoader(train_dataset, batch_size=32, shuffle=True)  # Create training data loader
val_loader = CustomDataLoader(val_dataset, batch_size=32)  # Create validation data loader

# Initialize model, criterion, and optimizer
model = CNN_Model()  # Initialize CNN model
criterion = nn.CrossEntropyLoss()  # Initialize cross-entropy loss criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initialize Adam optimizer with learning rate 0.001

# Create Trainer instance and start training
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader)  # Create Trainer instance with defined parameters
trainer.train()  # Start model training
trainer.save_model('trained_model.pth') # Saved model training

# Validate the model after training
trainer.validate()  # Validate the trained model on the validation dataset



"""
import os
import torch
from torch.nn import *
from Model import Model
from CustomDataLoader import CustomDataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
num_epochs = 5
# batch_size = 64
learning_rate = 0.005

model = Model().to(device)

# Loss and optimizer
criterion = MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# Train the model
path = 'training_set/'
labels = [0, 1, 2, 3, 4]
paths = ['training_set/' + x for x in os.listdir(path)]
train_loader = CustomDataLoader(paths=paths, labels=labels).construction(batch_size=1)
model = Model()
model.to(device)
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.double()

# Test the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Forward pass
        images, labels = data
        if images is not None:
            images = images.to(device)[None, :]
            labels = torch.Tensor(labels)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

torch.save(model.state_dict(), 'model.pth')
"""

