import gradio as gr
import os
import torch
from torch import nn
import torchvision
from timeit import default_timer as timer
from typing import Tuple, Dict


##############################################
# 1. Model Function
##############################################
def create_effnetb2_model(num_classes:int=9,
                          seed:int=42,
                          is_TrivialAugmentWide = True,
                          freeze_layers=True):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int): number of classes in the classifier head, default = 10
        seed (int): random seed value, default = 42
        is_TrivialAugmentWide (boolean): Artificially increase the diversity of a training dataset with data augmentation, default = True.

    Returns:
        effnetb2_model (torch.nn.Module): EfficientNet_B2 model.
        effnetb2_transforms (torchvision.transforms): EfficientNet_B2 image transforms.
    """
    # 1. Create EfficientNet_B2 pretrained weights and transforms
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2_transforms = weights.transforms()

    if is_TrivialAugmentWide:
        effnetb2_transforms = torchvision.transforms.Compose([
                  torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=8),
                  effnetb2_transforms,
      ])

    # 2. Create the EfficientNet_B2 model
    effnetb2_model = torchvision.models.efficientnet_b2(weights=weights)

    # 3. Freeze all layers of the model
    if freeze_layers:
        for param in effnetb2_model.parameters():
            param.requires_grad = False

    # 4. Change classifier head
    torch.manual_seed(seed)
    effnetb2_model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )

    return effnetb2_model, effnetb2_transforms
##############################################
# 1. Setup class names
##############################################
class_names = ['BasilicaDiSanPietro',
               'CreazioneDiAdamo-Pannello1',
               'CreazioneDiAdamo-Pannello2',
               'CreazioneDiAdamo-Pannello3',
               'CreazioneDiAdamo-Pannello4',
               'CreazioneDiEva-Pannello1',
               'CreazioneDiEva-Pannello2',
               'CreazioneDiEva-Pannello3',
               'CreazioneDiEva-Pannello4']

##############################################
# 2. Model and transforms preparation 
##############################################

# 2.1 Create EfficientNet_B2 model
EfficientNetB2_model, EfficientNetB2_transforms = create_effnetb2_model(num_classes=9,is_TrivialAugmentWide=False)

# 2.2 Load saved weights (from our trained PyTorch model)
EfficientNetB2_model.load_state_dict(
    torch.load(
        f="EfficientNet_B2_FT.pth",
        map_location=torch.device("cpu"),  # load to CPU because we will use the free HuggingFace Space CPUs.
    )
)

##############################################
# 3. Create prediction function
##############################################
def prediction(img) -> Tuple[Dict, float]:
    """returns prediction probabilities and prediction time.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = EfficientNetB2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    EfficientNetB2_model.eval()
    with torch.inference_mode():
        # Get prediction probabilities
        pred_probs = torch.softmax(EfficientNetB2_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class.
    # This is the required format for Gradio's output parameter.
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    #return pred_labels_and_probs, pred_time
    return max(pred_labels_and_probs, key=pred_labels_and_probs.get)
    
##############################################
# 4. Gradio app
##############################################
# 4.1 Create title, description and article strings
title = "Artwork Classification 🎨"
description = "An EfficientNetB2 computer vision model to classify artworks."
article = "Created with PyTorch."

# 4.2 Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# 4.3 Create the Gradio demo
demo = gr.Interface(fn=prediction, # mapping function from input to output
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # 1st output: pred_probs
                             gr.Number(label="Prediction time (s)")], # 2nd output: pred_time
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# 4.4 Launch the Gradio demo!
demo.launch()