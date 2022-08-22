import torch.nn as nn
from torchsummary import summary
from .base_classes import Conv, Lin


class FoodNet(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()
        config["architecture"]["layers"]["output"]["outC"] = num_classes
        self.net = self._build_model_from_config(config)
    
    @staticmethod
    def _build_model_from_config(config):
        network = nn.Sequential()
        for name, conf in config["architecture"]["layers"].items():
            if name.startswith("conv"):
                network.append(Conv(**conf))
            elif name.startswith("dropout"):
                network.append(nn.Dropout(conf))
            elif name.startswith("lin"):
                network.append(Lin(**conf))
            elif name.startswith("output"):
                network.append(Lin(**conf))
            elif name == "flatten":
                network.append(nn.Flatten())
            else:
                raise NotImplementedError

        return network

    def forward(self, x):
        return self.net(x)
