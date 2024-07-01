import gc

import torch
from ts.torch_handler.image_classifier import ImageClassifier


class CustomImageClassifier(ImageClassifier):
    def initialize(self, context):
        super().initialize(context)
        self.context = context

    def preprocess(self, data):
        # Use the parent class's preprocess method
        return super().preprocess(data)

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

# Save this code as custom_handler.py and use it in your model archive.
