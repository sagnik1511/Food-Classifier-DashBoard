import torch.nn as nn


class Conv(nn.Module):

    def __init__(self, inC, outC, kSize=3, stride=1, padding=0, eps=0.1):
        super().__init__()
        self._cnn = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=kSize, stride=stride, padding=padding),
            nn.BatchNorm2d(outC, eps=eps),
            nn.ReLU(),
        )

    def forward(self, x):
        return self._cnn(x)

class Lin(nn.Module):
    
    def __init__(self, inC, outC):
        super().__init__()
        self._linear = nn.Sequential(
            nn.Linear(inC, outC),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self._linear(x)