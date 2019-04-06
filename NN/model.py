import torch.nn as nn
import torch.nn.functional as F
import torch


class mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


