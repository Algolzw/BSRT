import torch
import torch.nn as nn
import torch.nn.functional as F

class Filter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        kernel = torch.tensor([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
        self.conv = nn.Conv2d(args.n_colors, args.n_colors, 3, 3)
        with torch.no_grad():
            self.conv.weight.copy_(kernel.float())
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        preds_x = self.conv(x)
        preds_y = self.conv(y)

        return self.loss(preds_x, preds_y)
