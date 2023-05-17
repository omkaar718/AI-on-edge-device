# efficientnet modified - remove classification head and add a conv to make the number of channels in output = 1
  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetWithConv(nn.Module):
    def __init__(self):
        super(EfficientNetWithConv, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
        self.efficientnet = models.efficientnet_b0(weights=weights)
        # Load the EfficientNetB0WithConv model

        self.efficientnet = self.efficientnet.features
        # Add a convolutional layer at the end of the features
        self.conv = nn.Conv2d(1280, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        x = self.efficientnet(x)
        #print('\nShape of x before conv: ', type(x), x.size())
        x = self.conv(x)

        return x
