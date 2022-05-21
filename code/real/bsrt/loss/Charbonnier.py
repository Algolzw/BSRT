import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """L1 charbonnier loss."""

    def __init__(self, epsilon=1e-3, reduce=True):
        super(CharbonnierLoss, self).__init__()
        self.eps = epsilon * epsilon
        self.reduce = reduce

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        if self.reduce:
            loss = torch.mean(error)
        else:
            loss = error
        return loss