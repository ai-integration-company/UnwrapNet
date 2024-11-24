import torch
from torch import nn
from torchvision import models


def build_custom_resnet18(in_channels):
    model = models.resnet18(pretrained=True)

    original_conv = model.conv1

    # Create a new conv layer with 9 input channels
    new_conv = nn.Conv2d(in_channels, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)

    # Initialize the new conv layer's weights by copying the original weights and duplicating for the additional channels
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = original_conv.weight
        new_conv.weight[:, 3:6, :, :] = original_conv.weight  # Duplicate weights for the additional channels
        new_conv.weight[:, 6:9, :, :] = original_conv.weight
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


class EfficientNetRegression(nn.Module):
    def __init__(self):
        super(EfficientNetRegression, self).__init__()
        self.efficient_net = models.efficientnet_b1(pretrained=True)

        # Modify the first convolutional layer to accept 6 channels
        original_conv = self.efficient_net.features[0][0]
        self.efficient_net.features[0][0] = nn.Conv2d(
            in_channels=9,  
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Modify the classifier to output a single value for regression
        self.efficient_net.classifier[1] = nn.Linear(
            self.efficient_net.classifier[1].in_features, 1
        )

    def forward(self, x):
        return self.efficient_net(x)
