import torch
import torch.nn as nn
import torch.optim as optim

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        # Define layers
        self.flatten = nn.Flatten()  # To flatten the input from (6, 5, 1) to (30,)
        
        # Hidden layers (Deep network with 5 layers)
        self.fc1 = nn.Linear(30, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 512) # Second hidden layer
        self.fc3 = nn.Linear(512, 1024) # Third hidden layer
        self.fc4 = nn.Linear(1024, 512) # Fourth hidden layer
        self.fc5 = nn.Linear(512, 256) # Fifth hidden layer
        
        # Output layer
        self.fc6 = nn.Linear(256, 1500)  # Output layer for flattened output
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)  # No activation at output
        return x




class DecoderNet(nn.Module):
    def __init__(self, inner_dims, seq_len=1):
        super().__init__() 
        self.inner_dims = inner_dims
        
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(seq_len, inner_dims[0], kernel_size=(2), stride=1, padding=0),  # [c,7,6]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[0], inner_dims[1], kernel_size=(3), stride=1, padding=0),       # [c,9,8]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[1], inner_dims[2], kernel_size=(3), stride=1, padding=0),       # [c,12,11]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[2], inner_dims[3], kernel_size=(3), stride=1, padding=(0)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[3], inner_dims[4], kernel_size=(4), stride=1, padding=(0,1)),      # [c,15,12]
            nn.ReLU(),
            nn.ConvTranspose2d(inner_dims[4], 1, kernel_size=(4,3), stride=2, padding=(2,1)),     # [c,30,25]
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded