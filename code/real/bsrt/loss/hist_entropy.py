import torch
import torch.nn as nn
import torch.nn.functional as F

class HistEntropy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        p = torch.softmax(x, dim=1)
        logp = torch.log_softmax(x, dim=1)

        entropy = (-p * logp).sum(dim=(2, 3)).mean()

        return entropy
