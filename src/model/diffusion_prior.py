import torch
import torch.nn as nn

class DiffusionPrior(nn.Module):
    def __init__(self, hidden : int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden,3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 3, padding=1),
        )

    def forward(self, x, t ):
        # T is purposly ignoored but added to keep compatability with api
        return self.net(x)