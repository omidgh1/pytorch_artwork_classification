import torch

class Predictor:
    """
    Predictor is a utility class for making predictions using a trained neural network model.
    Attributes:
    - model: The trained neural network model used for making predictions.
    - device: The device (CPU or GPU) on which the model should run inference.
    Methods:
    - predict(): Makes predictions on input data using the trained model.
    """

    def __init__(self, model, device='cpu'):
        """
        Initializes the Predictor object.
        Args:
            model (torch.nn.Module): The trained neural network model.
            device (str): The device to run inference on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)  # Move model to specified device

    def predict(self, input_data):
        """
        Makes predictions on input data using the trained model.
        Args:
            input_data (torch.Tensor): Input data for making predictions.
        Returns:
            predictions (torch.Tensor): Predicted output from the model.
        """
        with torch.no_grad():
            input_data = input_data.to(self.device)  # Move input data to specified device
            self.model.eval()  # Set model to evaluation mode
            predictions = self.model(input_data)  # Make predictions
        return predictions