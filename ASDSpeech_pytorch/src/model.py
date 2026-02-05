import torch
import torch.nn as nn

class ASDSpeechCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input will be (batch, 100, 49)
        self.conv1 = nn.Conv1d(in_channels=49, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # We will infer the size dynamically later
        self.fc = nn.Linear(64 * 49, 1)

    def forward(self, x):
        # x: (batch, 100, 49)
        x = x.permute(0, 2, 1)   # → (batch, 49, 100)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
