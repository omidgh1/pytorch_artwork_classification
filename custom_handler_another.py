import gc
import base64
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from ts.torch_handler.image_classifier import ImageClassifier


class CustomImageClassifier(ImageClassifier):
    def initialize(self, context):
        super().initialize(context)
        self.context = context

    #def preprocess(self, data):
        # Use the parent class's preprocess method
    #    return super().preprocess(data)
    def preprocess_image(self, image):
        # Implement any custom preprocessing logic here
        # Example: resizing, normalization, etc.
        return image  # This assumes your model expects PIL images
    def preprocess(self, data):
        # Ensure data is parsed correctly
        if isinstance(data, list):
            # Handle list format (if TorchServe sends data as a list)
            data = data[0]  # Assuming data is wrapped in a list

        # Handle base64 decoding and convert to PIL image
        base64_image = data['image']
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Apply transformations (resize, normalize, convert to tensor)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Example normalization parameters
                                 std=[0.229, 0.224, 0.225])
        ])
        tensor_image = preprocess(image).unsqueeze(0)  # Add batch dimension

        return tensor_image.to(self.device)

    def postprocess(self, data):
        # Use the parent class's postprocess method
        return super().postprocess(data)

    def handle(self, data, context):
        # Preprocess the input data
        preprocessed_data = self.preprocess(data)

        # Ensure the model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Perform the inference
            output = self.model(preprocessed_data)

        result = self.postprocess(output)

        # Clean up
        del preprocessed_data
        del output
        gc.collect()

        # Postprocess the output data
        return result