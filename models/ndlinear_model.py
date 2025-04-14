import torch.nn as nn
import torch.nn.functional as F
from ndlinear import NdLinear

class NdLinearCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.ndlinear = NdLinear(input_dims=(8, 8, 32), hidden_size=(5, 5, 20))
        self.dropout = nn.Dropout(0.3)
        self.final_fc = nn.Linear(500, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ndlinear(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.final_fc(x)