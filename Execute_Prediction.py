from PIL import Image
import torch
import torchvision.transforms as transforms
from Model import CNN_Model
from Prediction import Predictor  # Assuming you have the Predictor class defined in a separate file
import torch.nn.functional as F


# Load the saved model
saved_model_path = 'trained_model.pth'
model = CNN_Model()  # Assuming CNN_Model is your model architecture
model.load_state_dict(torch.load(saved_model_path))

# Assuming 'model' is your trained neural network model
predictor = Predictor(model)

# Open and preprocess the image
image_path = 'Data_backup/training_set/0/i - 154.jpeg'
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model's input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])
input_data = preprocess(image).unsqueeze(0)  # Add batch dimension

# Make predictions
predictions = predictor.predict(input_data)

# Assuming your model outputs logits, you may want to apply softmax to obtain probabilities
probabilities = F.softmax(predictions, dim=1)

# Assuming you want to get the class with the highest probability as the predicted label
predicted_label = torch.argmax(probabilities, dim=1).item()

# Print predicted label
print(f"Predicted label: {predicted_label}")