"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
from torchvision import models
from typing import Optional

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x


def create_model(name: str, num_classes: int, pretrained: bool = False, hidden_units: int = 64) -> nn.Module:
    """Factory to create models by name.

    Supported names: 'tinyvgg', 'resnet18', 'efficientnet_b0', 'mobilenet_v3_small'
    """
    name = name.lower()
    if name == 'tinyvgg':
        return TinyVGG(input_shape=3, hidden_units=hidden_units, output_shape=num_classes)

    if name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m

    if name == 'efficientnet_b0':
        m = models.efficientnet_b0(pretrained=pretrained)
        # classifier is (Dropout, Linear)
        if hasattr(m, 'classifier'):
            try:
                in_f = m.classifier[1].in_features
                m.classifier[1] = nn.Linear(in_f, num_classes)
            except Exception:
                # fallback: replace whole classifier
                m.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_f, num_classes))
        return m

    if name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(pretrained=pretrained)
        # classifier is usually Sequential(..., Linear)
        try:
            # find last linear layer
            if isinstance(m.classifier, nn.Sequential):
                last = list(m.classifier.children())[-1]
                in_f = last.in_features
                # replace last
                children = list(m.classifier.children())
                children[-1] = nn.Linear(in_f, num_classes)
                m.classifier = nn.Sequential(*children)
            else:
                # fallback
                m.classifier = nn.Sequential(nn.Flatten(), nn.Linear(m.classifier.in_features, num_classes))
        except Exception:
            pass
        return m

    raise ValueError(f"Unsupported model name: {name}")
