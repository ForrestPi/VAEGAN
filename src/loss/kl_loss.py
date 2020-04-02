import torch
import torch.nn as nn


class KLLoss(nn.Module):
    def __init__(self, size_average=False):
        super().__init__()
        self.size_average = size_average

    def forward(self, mu, logvar):
        loss = 0.5*(mu.pow(2) + logvar.exp() - logvar - 1)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
