import os
import torch
from torch.nn import *
from Model import Model
from CustomDataLoader import CustomDataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
num_epochs = 1
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
print('Ciao')
