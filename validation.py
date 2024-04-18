from Model import Model
from CustomDataLoader import CustomDataLoader
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()

model.load_state_dict(torch.load('model.pth'))
model.to(device)

#Evaluation
validation_path = 'validation_set/'
validation_labels = [0, 1, 2, 3, 4]
validation_paths = ['validation_set/' + x for x in os.listdir(validation_path)]
validation_loader = CustomDataLoader(paths=validation_paths, labels=validation_labels).construction(batch_size=1)
model.eval()

# Initialize variables for accuracy calculation
total_correct = 0
total_samples = 0

# Disable gradient calculation to speed up computation
with torch.no_grad():
    for data in validation_loader:
        images, labels = data
        if images is not None:
            images = images.to(device)[None, :]
            labels = torch.Tensor(labels).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the index of the maximum value as the predicted class
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = total_correct / total_samples
print('Accuracy on validation set: {:.2f}%'.format(accuracy * 100))