import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.features[0][0].stride = (1, 1)
    model.classifier[1] = nn.Linear(1280, 10)
    return model
