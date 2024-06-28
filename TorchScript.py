import torch
import torch.nn as nn
from torch import nn
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile


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

# Load your trained model
#model = EfficientNet()  # Replace with your model class
EfficientNetB2_model, EfficientNetB2_transforms = create_effnetb2_model(num_classes=9,is_TrivialAugmentWide=False)
EfficientNetB2_model.load_state_dict(
    torch.load(
        f="EfficientNet_B2_FT.pth",
        map_location=torch.device("cpu"),  # load to CPU because we will use the free HuggingFace Space CPUs.
    )
)
EfficientNetB2_model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(EfficientNetB2_model)
scripted_model.save('model_scripted.pt')

scripted_module = torch.jit.script(EfficientNetB2_model)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("my_model_lite.ptl")
