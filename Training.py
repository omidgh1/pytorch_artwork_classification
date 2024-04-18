from tqdm import tqdm
import torch

class Trainer:
    """
    Trainer is a utility class designed to facilitate the training and validation of neural network models in PyTorch.
    It encapsulates the process of training a model by iterating over a specified number of epochs, computing loss,
     and updating model parameters using backpropagation. Additionally, it provides a method for validating
     the trained model on a separate validation dataset to evaluate its performance.
    Attributes:
    - model: The neural network model to be trained.
    - criterion: The loss function used to compute the loss between predicted and target values.
    - optimizer: The optimization algorithm used to update the model parameters based on computed gradients.
    - train_loader: The DataLoader object containing the training dataset.
    - val_loader: The DataLoader object containing the validation dataset.
    - num_epochs: The number of epochs for training (default is 10).

    Methods:
    - train(): Executes the training loop over the specified number of epochs. It iterates over batches of data from the training DataLoader, computes the loss, and updates the model parameters using backpropagation.
    - validate(): Evaluates the trained model on the validation dataset to calculate the accuracy of predictions. It computes the accuracy by comparing predicted labels with ground truth labels and prints the validation accuracy.
    - save_model(): Saves the trained model to a file.
    """
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for batch_idx, (images, labels) in pbar:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                pbar.set_description(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / (batch_idx + 1):.4f}')

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy}')

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)