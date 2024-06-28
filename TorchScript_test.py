import torch
from torchvision import transforms
from PIL import Image
import os
import glob

class_labels = [
    "BasilicaDiSanPietro",
    "CreazioneDiAdamo-Pannello1",
    "CreazioneDiAdamo-Pannello2",
    "CreazioneDiAdamo-Pannello3",
    "CreazioneDiAdamo-Pannello4",
    "CreazioneDiEva-Pannello1",
    "CreazioneDiEva-Pannello2",
    "CreazioneDiEva-Pannello3",
    "CreazioneDiEva-Pannello4"
]

# Load the model
model_scripted = torch.load('model_scripted.pt', map_location=torch.device('cpu'))  # Use 'cuda' if available for GPU inference
#model_scripted.eval()  # Set the model to evaluation mode

for label in class_labels:
    jpg_files = glob.glob(os.path.join(f'test_images\{label}', '*.JPG'))
    for pic in jpg_files:

        # Define transformations (if any)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model's expected input size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize if needed
        ])

        # Example: Load an image and apply transformations
        #image_path = 'test-image/CreazioneDiAdamo-Pannello1/IMG_6652.JPG'
        image = Image.open(pic)
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model_scripted(input_tensor)
            _, predicted = torch.max(output, 1)

        # Example: Print the predicted class label
        predicted_label = predicted.item()
        #print(f'Predicted Label Index: {predicted_label}')

        # Map predicted index to class label
        predicted_class = class_labels[predicted_label]
        print(f'Predicted Class: {predicted_class}',f'Actual Class: {label}')