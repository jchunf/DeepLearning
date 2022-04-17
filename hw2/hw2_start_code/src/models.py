from torchvision import models
import torch.nn as nn
from res_jcf import ultranet18


def model_A(num_classes):
    model_resnet = models.resnet18(pretrained=False)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes):
    ## your code here
    model = ultranet18(num_classes)
    return  model


def model_C(num_classes):
    ## your code here
    pass