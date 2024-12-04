"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 
import torch.nn.functional as F


def choose_model(model_type="VGG24", output_shape=30, device="cuda", input_shape=1):
  """Returns Model from model_type.
  Args:
  model_type(str): String for which model to load
  classes(int): An integer indicating number of classes.
  device(str): Device to be used (cuda/cpu)
  """
  if(model_type=="VGG24"):
    model = VGG24(output_shape=output_shape).to(device)
  elif(model_type=="CNN"):
    model = SensorCNN(output_shape=output_shape).to(device)
  elif(model_type=="VGGVariation"):
    model = VGGVariation(output_shape=output_shape).to(device)
  print(f"[INFO] Model (Type: {model_type}, Classes: {output_shape}, Device: {device}) loaded.")
  return model

class VGGVariation(nn.Module):
    """Creates the VGGVariation architecture.
    Args:
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int) -> None:
        FEATURE_MAP=[16,32]
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),  
          nn.ReLU(),
          nn.Conv2d(in_channels=FEATURE_MAP[0], out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[0], FEATURE_MAP[1], kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[1], kernel_size=3, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=FEATURE_MAP[1]*6*5, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion







#fully working 
class SensorCNN(nn.Module):
    """Creates the simple one block with 2 Convolution 1 Maxpool CNN architecture.
    Args:
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int):
        super(SensorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 12, 128) 
        self.fc2 = nn.Linear(128, output_shape) 

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x
    






class VGG24(nn.Module):
    """Creates the VGGVariation architecture based on input with at least 24*24 input.
    Args:
    input_shape(int): An integer indicating number of input channels (default 1 channel).
    output_shape(int): An integer indicating number of classes.
    """
    def __init__(self, output_shape: int,input_shape=1) -> None:
        FEATURE_MAP=[32,64,128]
        super().__init__()
        self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),  
          nn.BatchNorm2d(FEATURE_MAP[0]),
          nn.ReLU(),
          nn.Conv2d(in_channels=FEATURE_MAP[0], out_channels=FEATURE_MAP[0], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[0]),                    
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_2 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[0], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[1], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[1]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv_block_3 = nn.Sequential(
          nn.Conv2d(FEATURE_MAP[1], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.Conv2d(FEATURE_MAP[2], FEATURE_MAP[2], kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(FEATURE_MAP[2]),          
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Dropout(0.5),
          nn.Linear(in_features=FEATURE_MAP[2]*3*3, out_features=1024),#needs to be changed according to data
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(in_features=1024, out_features=1024),
          nn.ReLU(),
          nn.Linear(in_features=1024, out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        #print(x.shape)
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.conv_block_3(x)
       # print(x.shape)
        x = self.classifier(x)
       # print(x.shape)
        return x
        # return self.classifier(self.block_2(self.block_1(x))) # <- leverage the benefits of operator fusion




