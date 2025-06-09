import torch.nn as nn

class GasSensorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(GasSensorNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)