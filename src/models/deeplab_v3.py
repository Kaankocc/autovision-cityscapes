import segmentation_models_pytorch as smp
import torch.nn as nn

def build_deeplabv3_plus(n_classes=7):
    """
    Phase 2 Model: DeepLabV3+
    Backbone: ResNet50 (Pre-trained on ImageNet)
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",        # Stronger feature extraction
        encoder_weights="imagenet",     # Starts with knowledge of the world
        in_channels=3,                  # RGB input
        classes=n_classes,              # 7 categories
        activation=None                 # Raw logits for CrossEntropyLoss
    )
    return model