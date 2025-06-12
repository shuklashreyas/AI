import torch
import torch.nn as nn

class FF_Net(nn.Module):
    def __init__(self):
        super(FF_Net, self).__init__()
        # Define a simple 3-layer feedforward network
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Input x: [batch_size, 784]
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        # For classification with CrossEntropyLoss, raw logits are returned
        return logits
