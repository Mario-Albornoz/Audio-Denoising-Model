import torch.nn as nn

from src.model.conv_tasnet import ConvTasNet


class DenoiseSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ConvTasNet(N=256, L=20, B=256, P=3, X=8, R=2)

    def forward(self, noisy):
        return self.model(noisy)