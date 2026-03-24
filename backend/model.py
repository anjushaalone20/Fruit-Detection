import torch
import torch.nn as nn
from torchvision import models
import json

class FruitClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(FruitClassifier, self).__init__()
        # Use a lightweight MobileNetV2 for fast inference and training
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)
        
        # Replace the classifier head with our num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def save_classes(class_names, path="backend/classes.json"):
    with open(path, "w") as f:
        json.dump(class_names, f)

def load_classes(path="backend/classes.json"):
    with open(path, "r") as f:
        return json.load(f)
